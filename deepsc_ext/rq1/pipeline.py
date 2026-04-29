"""One-command orchestration for the RQ1 experiment pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from deepsc_ext.rq1.common import ensure_dir, parse_snrs, write_json


STAGES = ["generate_data", "convert_data", "train", "decode", "evaluate", "plot"]


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def _script(name: str) -> Path:
    """Return a script path in the repository."""
    return _repo_root() / "scripts" / name


def _run(command: List[str]) -> None:
    """Run one subprocess command and fail fast on errors."""
    print("Running: {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=str(_repo_root()), check=True)


def _snrs_arg(args: argparse.Namespace) -> str:
    """Return the active SNR string, applying quick-test defaults."""
    if args.quick_test:
        return "0,6"
    parse_snrs(args.snrs)
    return args.snrs


def _write_pipeline_config(args: argparse.Namespace) -> None:
    """Write a reproducibility config snapshot."""
    config_dir = ensure_dir(args.output_dir / "configs")
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload["python"] = args.python
    payload["argv"] = sys.argv
    write_json(config_dir / "rq1_config.json", payload)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run selected RQ1 pipeline stages."""
    if args.quick_test:
        args.train_size = 200
        args.valid_size = 50
        args.test_size = 50
        args.epochs = 1
    _write_pipeline_config(args)

    stages = STAGES if args.stage == "all" else [args.stage]
    data_dir = args.data_root / "deepsc_format"
    snrs = _snrs_arg(args)
    python = args.python

    if "generate_data" in stages:
        _run(
            [
                python,
                str(_script("rq1_generate_data.py")),
                "--output-dir",
                str(args.data_root),
                "--train-size",
                str(args.train_size),
                "--valid-size",
                str(args.valid_size),
                "--test-size",
                str(args.test_size),
                "--seed",
                str(args.seed),
            ]
        )

    if "convert_data" in stages:
        _run(
            [
                python,
                str(_script("rq1_convert_data.py")),
                "--input-dir",
                str(args.data_root),
                "--output-dir",
                str(data_dir),
                "--seed",
                str(args.seed),
            ]
        )

    if "train" in stages and not args.skip_train:
        _run(
            [
                python,
                str(_script("rq1_train_deepsc.py")),
                "--data-dir",
                str(data_dir),
                "--checkpoint-dir",
                str(args.output_dir / "checkpoints" / "full"),
                "--log-dir",
                str(args.output_dir / "logs" / "train_full"),
                "--channel",
                args.channel,
                "--train-snr",
                str(args.train_snr),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--train-with-mine",
            ]
        )
        _run(
            [
                python,
                str(_script("rq1_train_deepsc.py")),
                "--data-dir",
                str(data_dir),
                "--checkpoint-dir",
                str(args.output_dir / "checkpoints" / "no_mi"),
                "--log-dir",
                str(args.output_dir / "logs" / "train_no_mi"),
                "--channel",
                args.channel,
                "--train-snr",
                str(args.train_snr),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--seed",
                str(args.seed),
                "--no-train-with-mine",
            ]
        )

    if "decode" in stages:
        for method in ["full", "no_mi"]:
            _run(
                [
                    python,
                    str(_script("rq1_decode_snr.py")),
                    "--data-dir",
                    str(data_dir),
                    "--test-jsonl",
                    str(args.data_root / "test.jsonl"),
                    "--checkpoint-dir",
                    str(args.output_dir / "checkpoints" / method),
                    "--output-dir",
                    str(args.output_dir / "decoded" / method),
                    "--channel",
                    args.channel,
                    "--snrs",
                    snrs,
                    "--batch-size",
                    str(args.decode_batch_size),
                    "--seed",
                    str(args.seed),
                    "--method",
                    method,
                ]
            )

    if "evaluate" in stages:
        _run(
            [
                python,
                str(_script("rq1_evaluate_task_metrics.py")),
                "--decoded-dir",
                str(args.output_dir / "decoded"),
                "--output-dir",
                str(args.output_dir / "metrics"),
            ]
        )

    if "plot" in stages:
        _run(
            [
                python,
                str(_script("rq1_plot_results.py")),
                "--summary-csv",
                str(args.output_dir / "metrics" / "rq1_summary.csv"),
                "--output-dir",
                str(args.output_dir / "figures"),
            ]
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ1 pipeline CLI parser."""
    parser = argparse.ArgumentParser(description="Run the RQ1 task semantics experiment pipeline.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all"] + STAGES,
        help="Pipeline stage to run.",
    )
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--snrs", default="0,3,6,9,12,15,18")
    parser.add_argument("--output-dir", default="outputs/rq1_task_semantics", type=Path)
    parser.add_argument("--data-root", default="data/rq1_task_semantics", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--channel", default="AWGN")
    parser.add_argument("--train-snr", default=6.0, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--decode-batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--train-size", default=8000, type=int)
    parser.add_argument("--valid-size", default=1000, type=int)
    parser.add_argument("--test-size", default=1000, type=int)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the RQ1 pipeline from parsed CLI args."""
    run_pipeline(args)
