"""Shared utilities for the RQ2 symbol efficiency pipeline."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from deepsc_ext.rq1.common import format_snr, read_json


DEFAULT_SYMBOLS = [1, 2, 3, 4, 6, 8, 10]
DEFAULT_SNRS = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
DEFAULT_THRESHOLDS = [0.8, 0.9, 0.95]
DEFAULT_METHODS = ["full", "no_mi"]


def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated integer list."""
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    try:
        values = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid integer list: {}".format(value)) from exc
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    return values


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated float list."""
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("At least one float is required.")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid float list: {}".format(value)) from exc


def parse_methods(value: str) -> List[str]:
    """Parse and validate RQ2 method names."""
    methods = [item.strip() for item in str(value).split(",") if item.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("At least one method is required.")
    invalid = [method for method in methods if method not in DEFAULT_METHODS]
    if invalid:
        raise argparse.ArgumentTypeError("Unsupported methods: {}".format(",".join(invalid)))
    return methods


def list_to_csv(values: Sequence[object]) -> str:
    """Render a list for subprocess command arguments."""
    return ",".join(str(value) for value in values)


def normalize_negative_csv_args(argv: Sequence[str], option_names: Sequence[str]) -> List[str]:
    """Allow argparse options like ``--snrs -15,-12``.

    Argparse treats comma-separated negative values as option-looking tokens.
    Converting them to ``--snrs=-15,-12`` keeps both CLI styles working.
    """
    result: List[str] = []
    index = 0
    options = set(option_names)
    while index < len(argv):
        item = argv[index]
        if item in options and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-") and "," in value:
                result.append("{}={}".format(item, value))
                index += 2
                continue
        result.append(item)
        index += 1
    return result


def warning(message: str) -> None:
    """Print a warning without requiring the warnings module formatting."""
    print("WARNING: {}".format(message), file=sys.stderr)


def spw_dir_name(symbols_per_word: int) -> str:
    """Return the directory name for one symbols-per-word value."""
    return "spw_{}".format(int(symbols_per_word))


def snr_filename(snr: float) -> str:
    """Return the decoded filename for one SNR value."""
    return "snr_{}.jsonl".format(format_snr(float(snr)))


def latest_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """Return the latest TensorFlow checkpoint prefix under checkpoint_dir."""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    return tf.train.latest_checkpoint(str(checkpoint_dir))


def infer_symbols_per_word_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """Infer symbols_per_word from the channel encoder output dimension."""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    for name, shape in tf.train.list_variables(checkpoint_path):
        if name.endswith("channel_encoder/dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE"):
            if len(shape) != 2:
                return None
            dim = int(shape[1])
            if dim <= 0 or dim % 2 != 0:
                return None
            return dim // 2
    return None


def read_checkpoint_config(checkpoint_dir: Path) -> dict:
    """Read checkpoint_dir/config.json when present."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        return {}
    return read_json(config_path)


def validate_checkpoint_symbols(
    checkpoint_dir: Path,
    expected_symbols_per_word: int,
    strict: bool = False,
) -> Optional[str]:
    """Validate and return a checkpoint path for one symbols-per-word setting.

    Missing checkpoints can be skipped in non-strict mode. Symbol mismatches are
    always treated as errors because TensorFlow would otherwise fail later with a
    less useful variable-shape message.
    """
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        message = "No TensorFlow checkpoint found in {}".format(checkpoint_dir)
        if strict:
            raise FileNotFoundError(message)
        warning(message)
        return None

    config = read_checkpoint_config(checkpoint_dir)
    actual = config.get("symbols_per_word")
    inferred = infer_symbols_per_word_from_checkpoint(checkpoint_path)
    if actual is None:
        actual = inferred
        if actual is None:
            message = "Could not infer symbols_per_word from {}".format(checkpoint_path)
            if strict:
                raise ValueError(message)
            warning(message)
            return None
    elif inferred is not None and int(inferred) != int(actual):
        raise ValueError(
            "symbols_per_word mismatch inside {}: config has {}, checkpoint variables imply {}".format(
                checkpoint_dir,
                actual,
                inferred,
            )
        )
    actual = int(actual)
    expected = int(expected_symbols_per_word)
    if actual != expected:
        raise ValueError(
            "symbols_per_word mismatch for {}: expected {}, checkpoint has {}".format(
                checkpoint_dir,
                expected,
                actual,
            )
        )
    return checkpoint_path


def active_snrs(mode: str, snrs: Sequence[float], fixed_snr: float) -> List[float]:
    """Return the SNR list implied by the pipeline mode."""
    if mode == "fixed_snr":
        return [float(fixed_snr)]
    return [float(snr) for snr in snrs]


def maybe_int(value: float):
    """Render integer-valued floats as ints for JSONL/CSV readability."""
    return int(value) if float(value).is_integer() else float(value)


def selected_methods(methods: Optional[Iterable[str]]) -> Optional[set]:
    """Convert optional methods list to a set for filtering."""
    if methods is None:
        return None
    return {str(method) for method in methods}
