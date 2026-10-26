"""CLI: python -m judgebias {check,run,plot} --config config.yaml"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .check import run_check
from .config import ConfigError, load_config
from .plot import plot_summary
from .runner import run, summary_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="judgebias")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("check", "validate config and .env keys, then make one test call per judge"),
                        ("run", "run judges x biases, then plot"),
                        ("plot", "plot results/summary.csv")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default="config.yaml")
        sp.add_argument("--log-dir", help="directory for the log file (default: <results_dir>/logs)")
        if name == "check":
            sp.add_argument("--no-call", action="store_true", help="skip the live test calls")
        if name == "run":
            sp.add_argument("--judge", action="append", help="judge label (repeatable)")
            sp.add_argument("--bias", action="append", help="bias name (repeatable)")
    args = p.parse_args(argv)
    fmt, datefmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    for noisy in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        cfg = load_config(args.config)
        log_dir = Path(args.log_dir) if args.log_dir else cfg.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{args.cmd}_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")  # all INFO+ records, incl. warnings/errors
        fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(fh)
        logging.info("command=%s config=%s log_file=%s", args.cmd, args.config, log_path)
        if args.cmd == "check":
            return 0 if run_check(cfg, live=not args.no_call) else 1
        if args.cmd == "run":
            asyncio.run(run(cfg, args.judge, args.bias))
        summary = pd.read_csv(summary_path(cfg))
        out = plot_summary(cfg, summary, cfg.results_dir / "bias_grid.png")
        print(f"Wrote {out}")
        return 0
    except ConfigError as e:
        logging.error("Config error: %s", e)
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    except Exception:
        logging.exception("Unhandled error")  # traceback goes to the log file too
        raise


if __name__ == "__main__":
    sys.exit(main())
