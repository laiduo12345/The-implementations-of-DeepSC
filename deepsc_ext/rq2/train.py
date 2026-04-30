"""Training wrapper for RQ2 symbol efficiency experiments."""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from deepsc_ext.rq1.common import command_string, ensure_dir, write_json
from deepsc_ext.rq1.deepsc_runner import DEFAULT_MODEL_CONFIG, train_deepsc


class _Tee:
    """Write stream output to both the console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    """Convert CLI values to JSON-serializable values."""
    payload: Dict[str, object] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def _write_checkpoint_config(args: argparse.Namespace, argv: Optional[list] = None) -> Path:
    """Write the RQ2 checkpoint config before training starts."""
    ensure_dir(args.checkpoint_dir)
    payload = _jsonable_args(args)
    payload.update(
        {
            "command": command_string(argv or sys.argv),
            "method": args.method,
            "train_with_mine": args.method == "full",
            "symbols_per_word": int(args.symbols_per_word),
            "channel_symbol_dim": int(args.symbols_per_word) * 2,
        }
    )
    config_path = args.checkpoint_dir / "config.json"
    write_json(config_path, payload)
    return config_path


def train_rq2(args: argparse.Namespace, argv: Optional[list] = None) -> Dict[str, object]:
    """Train one RQ2 method/symbols-per-word checkpoint."""
    if args.method not in {"full", "no_mi"}:
        raise ValueError("--method must be either full or no_mi.")
    if int(args.symbols_per_word) <= 0:
        raise ValueError("--symbols-per-word must be positive.")

    if args.quick_test:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 200
        args.max_valid_samples = args.max_valid_samples or 50

    if args.log_dir is None:
        args.log_dir = args.checkpoint_dir / "logs"
    if args.resume and args.resume_checkpoint is None:
        args.resume_checkpoint = args.checkpoint_dir

    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.log_dir)
    _write_checkpoint_config(args, argv=argv)
    train_log = args.checkpoint_dir / "train.log"

    train_args = argparse.Namespace(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        channel=args.channel,
        train_snr=args.train_snr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        mine_learning_rate=args.mine_learning_rate,
        shuffle_size=args.shuffle_size,
        max_length=args.max_length,
        symbols_per_word=args.symbols_per_word,
        resume_checkpoint=args.resume_checkpoint,
        train_with_mine=args.method == "full",
        quick_test=False,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
    )

    with train_log.open("a", encoding="utf-8") as handle:
        tee_out = _Tee(sys.stdout, handle)
        tee_err = _Tee(sys.stderr, handle)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print("RQ2 training method={} symbols_per_word={}".format(args.method, args.symbols_per_word))
            result = train_deepsc(train_args, argv=argv or sys.argv)
            print("Final checkpoint: {}".format(result["final_checkpoint"]))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ2 training CLI parser."""
    parser = argparse.ArgumentParser(description="Train one DeepSC checkpoint for RQ2 symbol efficiency.")
    parser.add_argument("--data-dir", default="data/rq1_massive/deepsc_format", type=Path)
    parser.add_argument("--checkpoint-dir", default="outputs/rq2_symbol_efficiency/checkpoints/no_mi/spw_8", type=Path)
    parser.add_argument("--log-dir", default=None, type=Path)
    parser.add_argument("--method", choices=["full", "no_mi"], default="no_mi")
    parser.add_argument("--symbols-per-word", default=8, type=int)
    parser.add_argument("--channel", default="AWGN")
    parser.add_argument("--train-snr", default=6.0, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--learning-rate", default=5e-4, type=float)
    parser.add_argument("--mine-learning-rate", default=1e-3, type=float)
    parser.add_argument("--shuffle-size", default=DEFAULT_MODEL_CONFIG["shuffle_size"], type=int)
    parser.add_argument("--max-length", default=DEFAULT_MODEL_CONFIG["max_length"], type=int)
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-valid-samples", default=None, type=int)
    parser.add_argument("--resume-checkpoint", default=None, type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --checkpoint-dir.")
    parser.add_argument("--quick-test", action="store_true")
    return parser


def main(args: argparse.Namespace, argv: Optional[list] = None) -> Dict[str, object]:
    """Run RQ2 training from parsed CLI args."""
    return train_rq2(args, argv=argv)

