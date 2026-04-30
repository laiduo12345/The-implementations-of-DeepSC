"""Threshold summaries for RQ2 symbol efficiency results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from deepsc_ext.rq1.common import ensure_dir, format_snr
from deepsc_ext.rq2.common import DEFAULT_THRESHOLDS, list_to_csv, parse_float_list


THRESHOLD_FIELDS = [
    "method",
    "snr",
    "threshold",
    "min_symbols_per_word",
    "achieved_task_success_rate",
    "status",
]


def _load_summary(path: Path) -> List[Dict[str, object]]:
    """Load RQ2 summary CSV rows."""
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, object] = dict(row)
            parsed["symbols_per_word"] = int(row["symbols_per_word"])
            parsed["snr"] = format_snr(float(row["snr"]))
            parsed["task_success_rate"] = float(row["task_success_rate"])
            rows.append(parsed)
    if not rows:
        raise ValueError("No rows found in {}".format(path))
    return rows


def summarize_thresholds(summary_csv: Path, output_csv: Path, thresholds: Sequence[float]) -> Path:
    """Write minimal symbols-per-word needed to reach each task success threshold."""
    rows = _load_summary(summary_csv)
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["snr"]))].append(row)

    output_rows: List[Dict[str, object]] = []
    for method, snr in sorted(grouped.keys(), key=lambda item: (item[0], float(item[1]))):
        values = sorted(grouped[(method, snr)], key=lambda row: int(row["symbols_per_word"]))
        for threshold in thresholds:
            reached = None
            for row in values:
                if float(row["task_success_rate"]) >= float(threshold):
                    reached = row
                    break
            if reached is None:
                output_rows.append(
                    {
                        "method": method,
                        "snr": snr,
                        "threshold": threshold,
                        "min_symbols_per_word": "",
                        "achieved_task_success_rate": "",
                        "status": "not_reached",
                    }
                )
            else:
                output_rows.append(
                    {
                        "method": method,
                        "snr": snr,
                        "threshold": threshold,
                        "min_symbols_per_word": reached["symbols_per_word"],
                        "achieved_task_success_rate": reached["task_success_rate"],
                        "status": "reached",
                    }
                )

    ensure_dir(output_csv.parent)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=THRESHOLD_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print("Wrote {}".format(output_csv))
    return output_csv


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ2 threshold CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize RQ2 task-success symbol thresholds.")
    parser.add_argument("--summary-csv", default="outputs/rq2_symbol_efficiency/metrics/rq2_summary.csv", type=Path)
    parser.add_argument("--output-csv", default="outputs/rq2_symbol_efficiency/metrics/rq2_thresholds.csv", type=Path)
    parser.add_argument("--thresholds", default=list_to_csv(DEFAULT_THRESHOLDS))
    return parser


def main(args: argparse.Namespace) -> Path:
    """Run threshold summarization from parsed CLI args."""
    thresholds = parse_float_list(args.thresholds) if isinstance(args.thresholds, str) else args.thresholds
    return summarize_thresholds(args.summary_csv, args.output_csv, thresholds)

