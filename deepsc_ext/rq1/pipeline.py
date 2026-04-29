"""One-command orchestration for the RQ1 experiment pipeline."""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from deepsc_ext.rq1.common import ensure_dir, parse_snrs, write_json


STAGES = ["generate_data", "convert_data", "train", "decode", "evaluate", "plot"]
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class PipelineStep:
    """One executable pipeline step."""

    label: str
    command: List[str]


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def _script(name: str) -> Path:
    """Return a script path in the repository."""
    return _repo_root() / "scripts" / name


def _log(message: str) -> None:
    """Write a message without breaking an active tqdm bar."""
    try:
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        tqdm.write(message)
    except Exception:
        print(message, flush=True)


def _warning(message: str) -> None:
    """Write a yellow warning message."""
    _log("{}WARNING: {}{}".format(YELLOW, message, RESET))


def _error(message: str) -> None:
    """Write a red error message."""
    _log("{}ERROR: {}{}".format(RED, message, RESET))


def _run(command: List[str], label: str, tf_log_level: str) -> None:
    """Run one subprocess command and fail fast on errors."""
    _log("Running [{}]: {}".format(label, " ".join(command)))
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = tf_log_level
    try:
        subprocess.run(command, cwd=str(_repo_root()), check=True, env=env)
    except FileNotFoundError as exc:
        _error("Step '{}' could not start: {}".format(label, exc))
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        _error("Step '{}' failed with exit code {}.".format(label, exc.returncode))
        raise SystemExit(exc.returncode)


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


def _build_steps(args: argparse.Namespace, stages: List[str], data_dir: Path, snrs: str) -> List[PipelineStep]:
    """Build executable pipeline steps in the requested order."""
    steps: List[PipelineStep] = []
    python = args.python
    if "generate_data" in stages:
        steps.append(
            PipelineStep(
                "generate_data",
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
                ],
            )
        )

    if "convert_data" in stages:
        steps.append(
            PipelineStep(
                "convert_data",
                [
                    python,
                    str(_script("rq1_convert_data.py")),
                    "--input-dir",
                    str(args.data_root),
                    "--output-dir",
                    str(data_dir),
                    "--seed",
                    str(args.seed),
                ],
            )
        )

    if "train" in stages:
        if args.skip_train:
            _warning("Skipping train stage because --skip-train was set; decode requires existing checkpoints.")
        else:
            steps.append(
                PipelineStep(
                    "train_full",
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
                    ],
                )
            )
            steps.append(
                PipelineStep(
                    "train_no_mi",
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
                    ],
                )
            )

    if "decode" in stages:
        if "train" not in stages or args.skip_train:
            for method in ["full", "no_mi"]:
                checkpoint_dir = args.output_dir / "checkpoints" / method
                if not checkpoint_dir.exists():
                    _warning("Checkpoint directory '{}' does not exist before decode.".format(checkpoint_dir))
        for method in ["full", "no_mi"]:
            steps.append(
                PipelineStep(
                    "decode_{}".format(method),
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
                    ],
                )
            )

    if "evaluate" in stages:
        if "decode" not in stages and not (args.output_dir / "decoded").exists():
            _warning("Decoded directory '{}' does not exist before evaluation.".format(args.output_dir / "decoded"))
        steps.append(
            PipelineStep(
                "evaluate",
                [
                    python,
                    str(_script("rq1_evaluate_task_metrics.py")),
                    "--decoded-dir",
                    str(args.output_dir / "decoded"),
                    "--output-dir",
                    str(args.output_dir / "metrics"),
                ],
            )
        )

    if "plot" in stages:
        summary_csv = args.output_dir / "metrics" / "rq1_summary.csv"
        if "evaluate" not in stages and not summary_csv.exists():
            _warning("Summary CSV '{}' does not exist before plotting.".format(summary_csv))
        steps.append(
            PipelineStep(
                "plot",
                [
                    python,
                    str(_script("rq1_plot_results.py")),
                    "--summary-csv",
                    str(summary_csv),
                    "--output-dir",
                    str(args.output_dir / "figures"),
                ],
            )
        )
    return steps


def _run_steps(steps: List[PipelineStep], show_progress: bool, tf_log_level: str) -> None:
    """Run all steps, optionally showing a pipeline progress bar."""
    if not steps:
        _warning("No pipeline steps to run.")
        return
    if not show_progress:
        for step in steps:
            _run(step.command, step.label, tf_log_level)
        return
    try:
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        with tqdm(total=len(steps), desc="RQ1 pipeline", unit="step") as progress:
            for step in steps:
                progress.set_postfix_str(step.label)
                _run(step.command, step.label, tf_log_level)
                progress.update(1)
    except ImportError:
        _warning("tqdm is unavailable; running without a progress bar.")
        for step in steps:
            _run(step.command, step.label, tf_log_level)


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
    steps = _build_steps(args, stages, data_dir, snrs)
    _run_steps(steps, show_progress=not args.no_progress, tf_log_level=args.tf_log_level)


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
    parser.add_argument("--no-progress", action="store_true", help="Disable the rq1_run_pipeline progress bar.")
    parser.add_argument(
        "--tf-log-level",
        default="2",
        choices=["0", "1", "2", "3"],
        help="TensorFlow C++ log level for child processes: 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the RQ1 pipeline from parsed CLI args."""
    try:
        run_pipeline(args)
    except ValueError as exc:
        _error(str(exc))
        raise SystemExit(2)
