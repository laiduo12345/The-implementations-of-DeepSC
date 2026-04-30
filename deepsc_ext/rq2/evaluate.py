"""Metric evaluation for RQ2 decoded outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepsc_ext.rq1.common import ensure_dir, format_snr, read_json, read_jsonl
from deepsc_ext.rq1.metrics import _case_metrics, _mean, _metadata_from_rows, corpus_bleu, normalize_text
from deepsc_ext.rq2.common import (
    DEFAULT_METHODS,
    DEFAULT_SNRS,
    DEFAULT_SYMBOLS,
    list_to_csv,
    parse_float_list,
    parse_int_list,
    parse_methods,
    snr_filename,
    spw_dir_name,
    warning,
)


SUMMARY_FIELDS = [
    "method",
    "symbols_per_word",
    "snr",
    "num_samples",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "intent_accuracy",
    "slot_precision",
    "slot_recall",
    "slot_f1",
    "task_success_rate",
    "avg_sentence_len",
    "avg_channel_uses",
    "task_success_per_symbol",
]


CASE_FIELDS = [
    "id",
    "method",
    "symbols_per_word",
    "snr",
    "original_text",
    "decoded_text",
    "true_intent",
    "pred_intent",
    "intent_correct",
    "slot_precision",
    "slot_recall",
    "slot_f1",
    "task_success",
    "slots_json",
    "sentence_len",
    "channel_uses",
]


def _parse_spw_from_path(path: Path) -> Optional[int]:
    """Parse spw_N from a decoded file path."""
    for part in path.parts:
        if part.startswith("spw_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _parse_snr_from_path(path: Path) -> str:
    """Parse an SNR string from snr_*.jsonl."""
    stem = path.stem
    if stem.startswith("snr_"):
        return stem.split("_", 1)[1].replace("_", ".")
    return ""


def _group_key(row: Dict[str, Any], path: Path) -> Tuple[str, int, str]:
    """Return the RQ2 grouping key for one decoded row."""
    method = str(row.get("method") or path.parent.parent.name)
    symbols_per_word = int(row.get("symbols_per_word") or _parse_spw_from_path(path) or 0)
    snr_raw = row.get("snr", _parse_snr_from_path(path))
    try:
        snr = format_snr(float(snr_raw))
    except (TypeError, ValueError):
        snr = str(snr_raw)
    return method, symbols_per_word, snr


def _sentence_len(text: str) -> int:
    """Return the unpadded, no-special-token sentence length."""
    normalized = normalize_text(text)
    return len(normalized.split()) if normalized else 0


def _matches_filters(
    key: Tuple[str, int, str],
    methods: Optional[set],
    symbols: Optional[set],
    snrs: Optional[set],
) -> bool:
    """Return whether a decoded group key passes optional filters."""
    method, symbols_per_word, snr = key
    if methods is not None and method not in methods:
        return False
    if symbols is not None and symbols_per_word not in symbols:
        return False
    if snrs is not None and snr not in snrs:
        return False
    return True


def _warn_missing_expected(decoded_dir: Path, methods, symbols_list, snrs) -> None:
    """Warn for expected decoded files that are absent."""
    if not methods or not symbols_list or not snrs:
        return
    for method in methods:
        for symbols_per_word in symbols_list:
            for snr in snrs:
                path = decoded_dir / method / spw_dir_name(symbols_per_word) / snr_filename(float(snr))
                if not path.exists():
                    warning("Missing decoded file {}".format(path))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows to a UTF-8 CSV."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {}".format(path))


def evaluate_decoded(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Evaluate RQ2 decoded JSONL files and write summary/case CSVs."""
    ensure_dir(args.output_dir)

    method_filter = set(args.methods) if args.methods else None
    symbols_filter = set(args.symbols_list) if args.symbols_list else None
    snr_filter = {format_snr(float(snr)) for snr in args.snrs} if args.snrs else None
    _warn_missing_expected(args.decoded_dir, args.methods, args.symbols_list, args.snrs)

    decoded_files = sorted(args.decoded_dir.glob("**/snr_*.jsonl"))
    if not decoded_files:
        raise FileNotFoundError("No decoded snr_*.jsonl files found under {}".format(args.decoded_dir))

    source_rows: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in decoded_files:
        for row in read_jsonl(path):
            key = _group_key(row, path)
            if not _matches_filters(key, method_filter, symbols_filter, snr_filter):
                continue
            method, symbols_per_word, snr = key
            row["method"] = method
            row["symbols_per_word"] = symbols_per_word
            row["snr"] = snr
            row.setdefault("decoded_text", "")
            row.setdefault("slots", {})
            source_rows[key].append(row)

    if not source_rows:
        raise FileNotFoundError("No decoded rows matched the requested RQ2 filters.")

    if args.metadata_json and args.metadata_json.exists():
        metadata = read_json(args.metadata_json)
    else:
        metadata = _metadata_from_rows(source_rows)

    grouped_cases: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    case_rows: List[Dict[str, Any]] = []
    for key, raw_rows in source_rows.items():
        _, symbols_per_word, _ = key
        for row in raw_rows:
            case_row = _case_metrics(row, raw_rows, metadata=metadata)
            length = _sentence_len(case_row["original_text"])
            case_row["symbols_per_word"] = symbols_per_word
            case_row["sentence_len"] = length
            case_row["channel_uses"] = length * symbols_per_word
            case_row["slots_json"] = json.dumps(row.get("slots", {}), ensure_ascii=False, sort_keys=True)
            case_rows.append({field: case_row.get(field, "") for field in CASE_FIELDS})
            grouped_cases[key].append(case_row)

    summary_rows: List[Dict[str, Any]] = []
    sorted_keys = sorted(grouped_cases.keys(), key=lambda item: (item[0], item[1], float(item[2])))
    for key in sorted_keys:
        method, symbols_per_word, snr = key
        rows = grouped_cases[key]
        raw_rows = source_rows[key]
        references = [row.get("original_text", "") for row in raw_rows]
        hypotheses = [row.get("decoded_text", "") for row in raw_rows]
        avg_sentence_len = _mean(rows, "sentence_len")
        avg_channel_uses = avg_sentence_len * symbols_per_word
        task_success_rate = _mean(rows, "task_success")
        summary_rows.append(
            {
                "method": method,
                "symbols_per_word": symbols_per_word,
                "snr": snr,
                "num_samples": len(rows),
                "bleu_1": corpus_bleu(references, hypotheses, 1),
                "bleu_2": corpus_bleu(references, hypotheses, 2),
                "bleu_3": corpus_bleu(references, hypotheses, 3),
                "bleu_4": corpus_bleu(references, hypotheses, 4),
                "intent_accuracy": _mean(rows, "intent_correct"),
                "slot_precision": _mean(rows, "slot_precision"),
                "slot_recall": _mean(rows, "slot_recall"),
                "slot_f1": _mean(rows, "slot_f1"),
                "task_success_rate": task_success_rate,
                "avg_sentence_len": avg_sentence_len,
                "avg_channel_uses": avg_channel_uses,
                "task_success_per_symbol": task_success_rate / avg_channel_uses if avg_channel_uses > 0 else 0.0,
            }
        )

    summary_path = args.output_dir / "rq2_summary.csv"
    cases_path = args.output_dir / "rq2_cases.csv"
    fixed_path = args.output_dir / "rq2_fixed_snr_summary.csv"
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(cases_path, CASE_FIELDS, case_rows)
    fixed_snr = format_snr(float(args.fixed_snr))
    fixed_rows = [row for row in summary_rows if str(row["snr"]) == fixed_snr]
    _write_csv(fixed_path, SUMMARY_FIELDS, fixed_rows)
    return summary_path, cases_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ2 evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate RQ2 decoded JSONL files.")
    parser.add_argument("--decoded-dir", default="outputs/rq2_symbol_efficiency/decoded", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq2_symbol_efficiency/metrics", type=Path)
    parser.add_argument("--metadata-json", default="data/rq1_massive/deepsc_format/metadata.json", type=Path)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--symbols-list", default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--fixed-snr", default=0.0, type=float)
    return parser


def main(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Run RQ2 metric evaluation from parsed CLI args."""
    if isinstance(args.methods, str) and args.methods:
        args.methods = parse_methods(args.methods)
    if isinstance(args.symbols_list, str) and args.symbols_list:
        args.symbols_list = parse_int_list(args.symbols_list)
    if isinstance(args.snrs, str) and args.snrs:
        args.snrs = parse_float_list(args.snrs)
    return evaluate_decoded(args)
