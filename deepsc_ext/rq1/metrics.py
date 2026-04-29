"""Task-level metrics for RQ1 decoded DeepSC outputs."""

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from deepsc_ext.rq1.common import ensure_dir, read_jsonl
from deepsc_ext.rq1.data import candidate_values_by_slot, flat_candidate_values


INTENT_SLOT_NAMES = {
    "book_flight": ["from_city", "to_city", "date"],
    "check_weather": ["city", "date"],
    "set_alarm": ["time"],
    "play_music": ["song", "artist"],
}


SUMMARY_FIELDS = [
    "method",
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
]


CASE_FIELDS = [
    "id",
    "method",
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
]


def normalize_text(text: str) -> str:
    """Normalize text for rule matching and slot matching."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether normalized text contains a normalized phrase."""
    phrase = normalize_text(phrase)
    if not phrase:
        return False
    return re.search(r"(?<!\w){}(?!\w)".format(re.escape(phrase)), text) is not None


def classify_intent(decoded_text: str) -> str:
    """Classify intent from decoded text using maintainable keyword rules."""
    text = normalize_text(decoded_text)
    if any(_contains_phrase(text, keyword) for keyword in ["weather", "forecast", "rain"]):
        return "check_weather"
    if any(_contains_phrase(text, keyword) for keyword in ["alarm", "wake"]):
        return "set_alarm"
    if any(_contains_phrase(text, keyword) for keyword in ["play", "listen", "song", "music"]):
        return "play_music"
    if any(_contains_phrase(text, keyword) for keyword in ["flight", "ticket", "reserve", "book"]):
        return "book_flight"

    # Entity-only fallbacks keep the first version usable when function words are corrupted.
    values = set(find_candidate_values(decoded_text, flat_candidate_values()))
    city_values = {normalize_text(value) for value in candidate_values_by_slot()["city"]}
    time_values = {normalize_text(value) for value in candidate_values_by_slot()["time"]}
    song_values = {normalize_text(value) for value in candidate_values_by_slot()["song"]}
    artist_values = {normalize_text(value) for value in candidate_values_by_slot()["artist"]}
    if len(values & city_values) >= 2:
        return "book_flight"
    if values & time_values:
        return "set_alarm"
    if values & (song_values | artist_values):
        return "play_music"
    if values & city_values:
        return "check_weather"
    return "unknown"


def candidates_for_intent(intent: str) -> List[str]:
    """Return slot candidate values relevant to an intent ontology."""
    by_slot = candidate_values_by_slot()
    values = []
    seen = set()
    for slot_name in INTENT_SLOT_NAMES.get(intent, []):
        for value in by_slot.get(slot_name, []):
            normalized = normalize_text(value)
            if normalized not in seen:
                values.append(value)
                seen.add(normalized)
    return values


def find_candidate_values(text: str, candidates: Iterable[str]) -> List[str]:
    """Find candidate slot values present in decoded text."""
    normalized_text = normalize_text(text)
    matches = []
    seen = set()
    for candidate in sorted(candidates, key=lambda value: len(normalize_text(value)), reverse=True):
        normalized = normalize_text(candidate)
        if normalized in seen:
            continue
        if _contains_phrase(normalized_text, normalized):
            matches.append(normalized)
            seen.add(normalized)
    return matches


def slot_metrics(decoded_text: str, true_intent: str, slots: Dict[str, str]) -> Tuple[float, float, float]:
    """Compute candidate-value slot precision, recall, and F1 for one sample."""
    gold_values = {normalize_text(value) for value in slots.values() if normalize_text(value)}
    predicted_values = set(find_candidate_values(decoded_text, candidates_for_intent(true_intent)))
    correct = len(gold_values & predicted_values)
    precision = correct / len(predicted_values) if predicted_values else 0.0
    recall = correct / len(gold_values) if gold_values else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _metric_tokens(text: str) -> List[str]:
    """Tokenize text for the local corpus BLEU implementation."""
    return normalize_text(text).split()


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter:
    """Count n-grams of one order."""
    return Counter(tuple(tokens[index : index + order]) for index in range(0, len(tokens) - order + 1))


def corpus_bleu(references: Sequence[str], hypotheses: Sequence[str], max_order: int) -> float:
    """Compute a simple zero-safe corpus BLEU score."""
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    reference_length = 0
    hypothesis_length = 0

    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = _metric_tokens(reference)
        hyp_tokens = _metric_tokens(hypothesis)
        reference_length += len(ref_tokens)
        hypothesis_length += len(hyp_tokens)
        for order in range(1, max_order + 1):
            ref_counts = _ngram_counts(ref_tokens, order)
            hyp_counts = _ngram_counts(hyp_tokens, order)
            overlap = hyp_counts & ref_counts
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(hyp_tokens) - order + 1, 0)

    if hypothesis_length == 0:
        return 0.0
    precisions = []
    for matches, possible in zip(matches_by_order, possible_by_order):
        if possible == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / possible)
    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(precision) for precision in precisions) / max_order)
    brevity_penalty = 1.0 if hypothesis_length > reference_length else math.exp(1.0 - reference_length / hypothesis_length)
    return float(geo_mean * brevity_penalty)


def _case_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all sample-level task metrics for one decoded row."""
    true_intent = row["intent"]
    pred_intent = classify_intent(row.get("decoded_text", ""))
    intent_correct = int(pred_intent == true_intent)
    precision, recall, f1 = slot_metrics(row.get("decoded_text", ""), true_intent, row.get("slots", {}))
    task_success = int(intent_correct == 1 and recall >= 1.0)
    return {
        "id": row.get("id", ""),
        "method": row.get("method", ""),
        "snr": row.get("snr", ""),
        "original_text": row.get("original_text", ""),
        "decoded_text": row.get("decoded_text", ""),
        "true_intent": true_intent,
        "pred_intent": pred_intent,
        "intent_correct": intent_correct,
        "slot_precision": precision,
        "slot_recall": recall,
        "slot_f1": f1,
        "task_success": task_success,
        "slots_json": json.dumps(row.get("slots", {}), ensure_ascii=False, sort_keys=True),
    }


def _mean(rows: Sequence[Dict[str, Any]], field: str) -> float:
    """Compute a mean for a numeric field."""
    if not rows:
        return 0.0
    return float(sum(float(row[field]) for row in rows) / len(rows))


def evaluate_decoded(decoded_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Evaluate every decoded JSONL file under decoded_dir and write CSV outputs."""
    ensure_dir(output_dir)
    decoded_files = sorted(decoded_dir.glob("**/snr_*.jsonl"))
    if not decoded_files:
        raise FileNotFoundError("No decoded snr_*.jsonl files found under {}".format(decoded_dir))

    case_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    source_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in decoded_files:
        for row in read_jsonl(path):
            method = row.get("method") or path.parent.name
            snr = str(row.get("snr", ""))
            row["method"] = method
            key = (method, snr)
            source_rows[key].append(row)
            case_row = _case_metrics(row)
            case_rows.append(case_row)
            grouped[key].append(case_row)

    summary_rows = []
    for key in sorted(grouped.keys(), key=lambda item: (item[0], float(item[1]))):
        method, snr = key
        rows = grouped[key]
        raw_rows = source_rows[key]
        references = [row.get("original_text", "") for row in raw_rows]
        hypotheses = [row.get("decoded_text", "") for row in raw_rows]
        summary_rows.append(
            {
                "method": method,
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
                "task_success_rate": _mean(rows, "task_success"),
            }
        )

    summary_path = output_dir / "rq1_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    cases_path = output_dir / "rq1_cases.csv"
    with cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(case_rows)

    print("Wrote {}".format(summary_path))
    print("Wrote {}".format(cases_path))
    return summary_path, cases_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ1 metrics CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate RQ1 decoded JSONL files.")
    parser.add_argument("--decoded-dir", default="outputs/rq1_task_semantics/decoded", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq1_task_semantics/metrics", type=Path)
    return parser


def main(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Run metrics from parsed CLI args."""
    return evaluate_decoded(args.decoded_dir, args.output_dir)
