"""CLI: python -m judgebias {check,run,plot,compare,runs,get,snapshot,newcode} --config config.yaml"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .check import run_check
from .compare import compare_all
from . import registry
from .config import ConfigError, config_origin, load_config, snapshot_config
from .plot import plot_all
from .runner import given_output_dir, latest_output_dir, merge_details, merge_summaries, new_output_dir, run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="judgebias")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("check", "validate config and .env keys, then make one test call per judge"),
                        ("run", "run judges x biases, then plot"),
                        ("plot", "plot the latest run (or merge several with --run) into a new folder"),
                        ("compare", "compare finished runs side by side (no model calls) into a new folder"),
                        ("runs", "list run codes"),
                        ("get", "zip just the result files of the run with this code, for download"),
                        ("snapshot", "copy the config with absolute paths into a new run folder (or dest), for a job that reads it later"),
                        ("newcode", "print an unused run code (job scripts name the job after it, then pass it as --code)")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", default="config.yaml")
        sp.add_argument("--log-dir", help="directory for the log file (default: <results_dir>/logs)")
        if name in ("run", "plot", "compare", "snapshot"):
            sp.add_argument("--out-dir", help="write here (must be new or empty) instead of results_dir/<timestamp>")
        if name in ("run", "plot"):
            # Chosen by a job script (see newcode); registered now unless it already is, e.g. a vllm_split.sh code.
            sp.add_argument("--code", help=argparse.SUPPRESS)
        if name == "snapshot":
            sp.add_argument("dest", nargs="?", help="write here instead of <out-dir>/config.yaml (e.g. for a check job)")
        if name == "get":
            sp.add_argument("code")
            sp.add_argument("--dest", help="zip file to write (default: <results_dir>/exports/<code>.zip)")
        if name in ("check", "run"):
            sp.add_argument("--model", action="append",
                            help="only judges with this model ID (repeatable), e.g. the one a vLLM job serves")
        if name == "check":
            sp.add_argument("--no-call", action="store_true", help="skip the live test calls")
        if name == "run":
            sp.add_argument("--judge", action="append", help="judge label (repeatable)")
            sp.add_argument("--bias", action="append", help="bias name (repeatable)")
        if name == "compare":
            sp.add_argument("runs", nargs="+", metavar="RUN",
                            help="downloaded run: folder or zip (path or name under results_dir; a vllm_split.sh "
                                 "folder with merged/ and/or part*/ works too), or run code; each one's groups are models")
        if name == "plot":
            sp.add_argument("--run", action="append",
                            help="run code, or folder (path or name under results_dir), repeatable; later wins on overlap")
            sp.add_argument("--allow-missing", action="store_true",
                            help="skip --run folders without summary.csv (e.g. a failed part of a split job)")
    args = p.parse_args(argv)
    fmt, datefmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    for noisy in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_dir = None
    try:
        cfg = load_config(args.config)
        if args.cmd == "snapshot":
            dest = Path(args.dest) if args.dest else _out_dir(cfg, args) / "config.yaml"  # run/plot --out-dir it later
            print(snapshot_config(args.config, dest))
            return 0
        if args.cmd == "runs":
            return _list_runs(cfg)
        if args.cmd == "newcode":
            print(registry.new_code(cfg.results_dir))
            return 0
        if args.cmd == "get":
            dest = registry.export(cfg.results_dir, args.code, Path(args.dest) if args.dest else None)
            print(f"Wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return 0
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
        if args.cmd == "compare":
            sources = [_compare_source(cfg, name) for name in args.runs]
            out_dir = _out_dir(cfg, args)
            (out_dir / "sources.txt").write_text("".join(f"{d}\n" for d in sources), encoding="utf-8")
            logging.info("comparing %s -> %s", [d.name for d in sources], out_dir)
            for out in compare_all(sources, out_dir):
                print(f"Wrote {out}")
            return 0
        if args.cmd == "check":
            return 0 if run_check(cfg, live=not args.no_call) else 1
        if args.cmd == "run":
            out_dir = _out_dir(cfg, args)
            logging.info("output folder: %s", out_dir)
            code = _register(cfg, args, out_dir, f"models={','.join(sorted({j.model for j in cfg.judges}))} "
                                                 f"judges={','.join(args.judge or ['all'])} biases={','.join(args.bias or ['all'])}")
            summary = asyncio.run(run(cfg, out_dir, args.judge, args.bias, code))
        else:
            sources = _sources(cfg, args.run, args.allow_missing) if args.run else [latest_output_dir(cfg)]
            summary = merge_summaries(sources)
            out_dir = _out_dir(cfg, args)
            summary["code"] = _register(cfg, args, out_dir, "merged " + " ".join(d.name for d in sources))
            summary.to_csv(out_dir / "summary.csv", index=False)
            merge_details(sources, out_dir)
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


def _out_dir(cfg, args) -> Path:
    return given_output_dir(Path(args.out_dir)) if args.out_dir else new_output_dir(cfg)


def _register(cfg, args, out_dir: Path, note: str) -> str:
    """This result's run code: --code (registered here unless it already is, e.g. a vllm_split.sh job's) or a new one."""
    dest = out_dir / "config.yaml"  # the exact config this result came from; a job script may have put it there
    if not (dest.exists() and dest.samefile(args.config)):
        snapshot_config(args.config, dest)
    code = args.code.strip().lower() if args.code else None
    if code and registry.lookup(cfg.results_dir, code):
        logging.info("part of run code %s", code)
        return code
    job = os.environ.get("SLURM_JOB_ID")
    note = f"config={Path(config_origin(args.config)).name} {note}" + (f" job={job}" if job else "")
    code = registry.register(cfg.results_dir, out_dir, args.cmd, note, code=code)
    logging.info("run code: %s  (python -m judgebias get %s)", code, code)
    print(f"Run code: {code}", flush=True)  # stdout too: job.sh sends the log records to the .err file
    return code


def _list_runs(cfg) -> int:
    entries = registry.load(cfg.results_dir)
    if not entries:
        print(f"No runs registered in {registry.registry_path(cfg.results_dir)}")
    for r in entries:
        done = (registry.result_dir(cfg.results_dir, r) / "summary.csv").exists()
        print(f"{r['code']}  {r['created']}  {r['kind']:<5}  {'done' if done else 'pending':<7}  {r['folder']}  {r['note']}")
    return 0


def _sources(cfg, names: list[str], allow_missing: bool) -> list[Path]:
    sources = []
    for name in names:
        try:
            sources.append(_run_dir(cfg, name))
        except FileNotFoundError as e:
            if not allow_missing:
                raise
            logging.warning("%s; merging the other folders without it", e)
    if not sources:
        raise FileNotFoundError(f"None of the run folders {names} has a summary.csv")
    return sources


def _compare_source(cfg, name: str) -> Path:
    """A compare source as given (folder or zip), else under results_dir, else a registered run code (for a split:
    its folder, so the parts are read too); compare finds the summary inside."""
    for path in (Path(name), cfg.results_dir / name):
        if path.exists():
            return path
    if entry := registry.lookup(cfg.results_dir, name):
        folder = registry.result_dir(cfg.results_dir, entry)
        return folder.parent if entry["kind"] == "split" else folder
    raise FileNotFoundError(f"No run folder, zip or run code '{name}' (also looked in {cfg.results_dir})")


def _run_dir(cfg, name: str) -> Path:
    path = Path(name)
    if not (path / "summary.csv").exists():
        path = cfg.results_dir / name
    if not (path / "summary.csv").exists() and (entry := registry.lookup(cfg.results_dir, name)):
        path = registry.result_dir(cfg.results_dir, entry)
    if not (path / "summary.csv").exists():
        raise FileNotFoundError(f"No summary.csv in run folder '{name}'")
    return path


if __name__ == "__main__":
    sys.exit(main())
