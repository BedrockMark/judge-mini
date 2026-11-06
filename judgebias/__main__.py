"""CLI: python -m judgebias {check,run,plot} --config config.yaml"""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .check import run_check
from .config import ConfigError, load_config
from .plot import plot_all
from .runner import latest_output_dir, merge_summaries, new_output_dir, run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="judgebias")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("check", "validate config and .env keys, then make one test call per judge"),
                        ("run", "run judges x biases, then plot"),
                        ("plot", "plot the latest run (or merge several with --run) into a new folder")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default="config.yaml")
        sp.add_argument("--log-dir", help="directory for the log file (default: <results_dir>/logs)")
        if name in ("check", "run"):
            sp.add_argument("--model", action="append",
                            help="only judges with this model ID (repeatable), e.g. the one a vLLM job serves")
        if name == "check":
            sp.add_argument("--no-call", action="store_true", help="skip the live test calls")
        if name == "run":
            sp.add_argument("--judge", action="append", help="judge label (repeatable)")
            sp.add_argument("--bias", action="append", help="bias name (repeatable)")
        if name == "plot":
            sp.add_argument("--run", action="append",
                            help="run folder (path or name under results_dir), repeatable; later wins on overlap")
    args = p.parse_args(argv)
    fmt, datefmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    for noisy in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_dir = None
    try:
        cfg = load_config(args.config)
        if getattr(args, "model", None):
            cfg.judges = [j for j in cfg.judges if j.model in args.model]
            if not cfg.judges:
                raise ConfigError(f"No judge in {args.config} uses model {args.model}")
        log_dir = Path(args.log_dir) if args.log_dir else cfg.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{args.cmd}_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")  # all INFO+ records, incl. warnings/errors
        fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(fh)
        logging.getLogger().setLevel(logging.INFO)  # basicConfig is a no-op if the root logger already has handlers
        logging.info("command=%s config=%s log_file=%s", args.cmd, args.config, log_path)
        if args.cmd == "check":
            return 0 if run_check(cfg, live=not args.no_call) else 1
        if args.cmd == "run":
            out_dir = new_output_dir(cfg)
            logging.info("output folder: %s", out_dir)
            summary = asyncio.run(run(cfg, out_dir, args.judge, args.bias))
        else:
            sources = [_run_dir(cfg, r) for r in args.run] if args.run else [latest_output_dir(cfg)]
            summary = merge_summaries(sources)
            out_dir = new_output_dir(cfg)
            summary.to_csv(out_dir / "summary.csv", index=False)
            (out_dir / "sources.txt").write_text("".join(f"{d}\n" for d in sources), encoding="utf-8")
            logging.info("plotting %s -> %s", [d.name for d in sources], out_dir)
        for out in plot_all(cfg, summary, out_dir):
            print(f"Wrote {out}")
        return 0
    except ConfigError as e:
        logging.error("Config error: %s", e)
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except Exception:
        logging.exception("Unhandled error")  # traceback goes to the log file too
        raise
    finally:
        if out_dir is not None:  # copy this command's log next to its results (also when it failed)
            fh.flush()
            shutil.copy2(log_path, out_dir / log_path.name)


def _run_dir(cfg, name: str) -> Path:
    path = Path(name)
    if not (path / "summary.csv").exists():
        path = cfg.results_dir / name
    if not (path / "summary.csv").exists():
        raise FileNotFoundError(f"No summary.csv in run folder '{name}'")
    return path


if __name__ == "__main__":
    sys.exit(main())
