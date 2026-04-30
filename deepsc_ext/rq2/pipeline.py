"""One-command orchestration for the RQ2 symbol efficiency experiment."""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from deepsc_ext.rq1.common import ensure_dir, write_json
from deepsc_ext.rq2.common import (
    DEFAULT_METHODS,
    DEFAULT_SNRS,
    DEFAULT_SYMBOLS,
    DEFAULT_THRESHOLDS,
    active_snrs,
    list_to_csv,
    parse_float_list,
    parse_int_list,
    parse_methods,
    spw_dir_name,
    warning,
)


STAGES = ["train", "decode", "evaluate", "thresholds", "plot"]
TRAIN_MAX_ATTEMPTS = 5
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
    """Write a pipeline message."""
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


def _run(command: List[str], label: str, tf_log_level: str, max_attempts: int = 1) -> None:
    """Run one subprocess command with optional retries."""
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = tf_log_level
    attempt = 1
    while True:
        if max_attempts > 1:
            _log(
                "Running [{}] (attempt {}/{}): {}".format(
                    label, attempt, max_attempts, " ".join(command)
                )
            )
        else:
            _log("Running [{}]: {}".format(label, " ".join(command)))
        try:
            subprocess.run(command, cwd=str(_repo_root()), check=True, env=env)
            return
        except FileNotFoundError as exc:
            _error("Step '{}' could not start: {}".format(label, exc))
            if attempt >= max_attempts:
                raise SystemExit(1)
        except subprocess.CalledProcessError as exc:
            _error("Step '{}' failed with exit code {}.".format(label, exc.returncode))
            if attempt >= max_attempts:
                raise SystemExit(exc.returncode)
        attempt += 1
        _warning("Retrying step '{}' ({}/{})...".format(label, attempt, max_attempts))


def _run_steps(steps: List[PipelineStep], show_progress: bool, tf_log_level: str) -> None:
    """Run all requested pipeline steps."""
    if not steps:
        _warning("No pipeline steps to run.")
        return
    if not show_progress:
        for step in steps:
            max_attempts = TRAIN_MAX_ATTEMPTS if step.label.startswith("train") else 1
            _run(step.command, step.label, tf_log_level, max_attempts)
        return
    try:
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        with tqdm(total=len(steps), desc="RQ2 pipeline", unit="step") as progress:
            for step in steps:
                progress.set_postfix_str(step.label)
                max_attempts = TRAIN_MAX_ATTEMPTS if step.label.startswith("train") else 1
                _run(step.command, step.label, tf_log_level, max_attempts)
                progress.update(1)
    except ImportError:
        _warning("tqdm is unavailable; running without a progress bar.")
        for step in steps:
            max_attempts = TRAIN_MAX_ATTEMPTS if step.label.startswith("train") else 1
            _run(step.command, step.label, tf_log_level, max_attempts)


def _apply_quick_test(args: argparse.Namespace) -> None:
    """Apply quick-test defaults."""
    if not args.quick_test:
        return
    args.symbols_list = [1, 4]
    args.snrs = [-9.0, 0.0]
    args.methods = ["no_mi"]
    args.epochs = 1
    args.max_train_samples = 200
    args.max_valid_samples = 50
    args.max_test_samples = 100


def _write_pipeline_config(args: argparse.Namespace, snrs: List[float]) -> None:
    """Write a reproducibility config snapshot."""
    config_dir = ensure_dir(args.output_dir / "configs")
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload["active_snrs"] = snrs
    payload["python"] = args.python
    payload["argv"] = sys.argv
    write_json(config_dir / "rq2_config.json", payload)


def _build_train_steps(args: argparse.Namespace) -> List[PipelineStep]:
    """Build training subprocess steps."""
    if args.skip_train:
        _warning("Skipping train stage because --skip-train was set; decode requires existing checkpoints.")
        return []
    steps: List[PipelineStep] = []
    for method in args.methods:
        for symbols_per_word in args.symbols_list:
            checkpoint_dir = args.output_dir / "checkpoints" / method / spw_dir_name(symbols_per_word)
            command = [
                args.python,
                str(_script("rq2_train_deepsc.py")),
                "--data-dir",
                str(args.data_dir),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--log-dir",
                str(args.output_dir / "logs" / "train_{}_spw_{}".format(method, symbols_per_word)),
                "--method",
                method,
                "--symbols-per-word",
                str(symbols_per_word),
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
                "--max-length",
                str(args.max_length),
            ]
            if args.max_train_samples:
                command.extend(["--max-train-samples", str(args.max_train_samples)])
            if args.max_valid_samples:
                command.extend(["--max-valid-samples", str(args.max_valid_samples)])
            if args.resume:
                command.append("--resume")
            steps.append(PipelineStep("train_{}_spw_{}".format(method, symbols_per_word), command))
    return steps


def _build_steps(args: argparse.Namespace, stages: List[str], snrs: List[float]) -> List[PipelineStep]:
    """Build executable pipeline steps in the requested order."""
    steps: List[PipelineStep] = []
    if "train" in stages:
        steps.extend(_build_train_steps(args))

    if "decode" in stages:
        command = [
            args.python,
            str(_script("rq2_decode_grid.py")),
            "--data-dir",
            str(args.data_dir),
            "--test-jsonl",
            str(args.test_jsonl),
            "--checkpoint-root",
            str(args.output_dir / "checkpoints"),
            "--output-dir",
            str(args.output_dir / "decoded"),
            "--symbols-list={}".format(list_to_csv(args.symbols_list)),
            "--snrs={}".format(list_to_csv(args.snrs)),
            "--methods={}".format(list_to_csv(args.methods)),
            "--mode",
            args.mode,
            "--fixed-snr",
            str(args.fixed_snr),
            "--channel",
            args.channel,
            "--batch-size",
            str(args.eval_batch_size),
            "--seed",
            str(args.seed),
            "--max-length",
            str(args.max_length),
        ]
        if args.max_test_samples:
            command.extend(["--max-test-samples", str(args.max_test_samples)])
        if args.strict_checkpoints:
            command.append("--strict-checkpoints")
        steps.append(PipelineStep("decode", command))

    if "evaluate" in stages:
        steps.append(
            PipelineStep(
                "evaluate",
                [
                    args.python,
                    str(_script("rq2_evaluate_metrics.py")),
                    "--decoded-dir",
                    str(args.output_dir / "decoded"),
                    "--output-dir",
                    str(args.output_dir / "metrics"),
                    "--metadata-json",
                    str(args.data_dir / "metadata.json"),
                    "--methods={}".format(list_to_csv(args.methods)),
                    "--symbols-list={}".format(list_to_csv(args.symbols_list)),
                    "--snrs={}".format(list_to_csv(snrs)),
                    "--fixed-snr",
                    str(args.fixed_snr),
                ],
            )
        )

    if "thresholds" in stages:
        steps.append(
            PipelineStep(
                "thresholds",
                [
                    args.python,
                    str(_script("rq2_summarize_thresholds.py")),
                    "--summary-csv",
                    str(args.output_dir / "metrics" / "rq2_summary.csv"),
                    "--output-csv",
                    str(args.output_dir / "metrics" / "rq2_thresholds.csv"),
                    "--thresholds={}".format(list_to_csv(args.thresholds)),
                ],
            )
        )

    if "plot" in stages:
        command = [
            args.python,
            str(_script("rq2_plot_results.py")),
            "--summary-csv",
            str(args.output_dir / "metrics" / "rq2_summary.csv"),
            "--thresholds-csv",
            str(args.output_dir / "metrics" / "rq2_thresholds.csv"),
            "--output-dir",
            str(args.output_dir / "figures"),
        ]
        if args.save_pdf:
            command.append("--save-pdf")
        steps.append(PipelineStep("plot", command))
    return steps


def run_pipeline(args: argparse.Namespace) -> None:
    """Run selected RQ2 pipeline stages."""
    _apply_quick_test(args)
    snrs = active_snrs(args.mode, args.snrs, args.fixed_snr)
    _write_pipeline_config(args, snrs)

    stages = STAGES if args.stage == "all" else [args.stage]
    steps = _build_steps(args, stages, snrs)
    _run_steps(steps, show_progress=not args.no_progress, tf_log_level=args.tf_log_level)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ2 pipeline CLI parser."""
    parser = argparse.ArgumentParser(description="Run the RQ2 symbol efficiency experiment pipeline.")
    parser.add_argument("--stage", default="all", choices=["all"] + STAGES)
    parser.add_argument("--data-dir", default="data/rq1_massive/deepsc_format", type=Path)
    parser.add_argument("--test-jsonl", default="data/rq1_massive/test.jsonl", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq2_symbol_efficiency", type=Path)
    parser.add_argument("--symbols-list", default=list_to_csv(DEFAULT_SYMBOLS))
    parser.add_argument("--snrs", default=list_to_csv(DEFAULT_SNRS))
    parser.add_argument("--methods", default=list_to_csv(DEFAULT_METHODS))
    parser.add_argument("--mode", default="grid", choices=["grid", "fixed_snr"])
    parser.add_argument("--fixed-snr", default=0.0, type=float)
    parser.add_argument("--thresholds", default=list_to_csv(DEFAULT_THRESHOLDS))
    parser.add_argument("--train-snr", default=6.0, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-size", "--decode-batch-size", dest="eval_batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--channel", default="AWGN")
    parser.add_argument("--max-length", default=35, type=int)
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-valid-samples", default=None, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-checkpoints", action="store_true")
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--tf-log-level",
        default="2",
        choices=["0", "1", "2", "3"],
        help="TensorFlow C++ log level for child processes.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Run the RQ2 pipeline from parsed CLI args."""
    try:
        if isinstance(args.symbols_list, str):
            args.symbols_list = parse_int_list(args.symbols_list)
        if isinstance(args.snrs, str):
            args.snrs = parse_float_list(args.snrs)
        if isinstance(args.methods, str):
            args.methods = parse_methods(args.methods)
        if isinstance(args.thresholds, str):
            args.thresholds = parse_float_list(args.thresholds)
        run_pipeline(args)
    except ValueError as exc:
        _error(str(exc))
        raise SystemExit(2)

