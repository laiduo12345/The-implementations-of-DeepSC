"""Shared utilities for the RQ1 task semantics pipeline."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a UTF-8 JSONL file."""
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to a UTF-8 JSONL file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with stable formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow if TensorFlow is imported."""
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        tf.random.set_seed(seed)
    except Exception:
        pass


def parse_snrs(snrs: str) -> List[float]:
    """Parse comma-separated SNR values."""
    values = []
    for item in snrs.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one SNR value is required.")
    return values


def format_snr(snr: float) -> str:
    """Format an SNR value for filenames and CSV output."""
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr).replace(".", "_")


def snr_to_noise(snr: float) -> float:
    """Convert SNR in dB to the noise std used by the original DeepSC code."""
    linear = 10 ** (float(snr) / 10.0)
    return float(1.0 / np.sqrt(2.0 * linear))


def command_string(argv: List[str]) -> str:
    """Return a readable command string for experiment logs."""
    return " ".join(argv)
