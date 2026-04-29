"""Import Amazon MASSIVE data into the RQ1 JSONL schema."""

import argparse
import json
import random
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from deepsc_ext.rq1.common import ensure_dir, write_json, write_jsonl


DEFAULT_MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
PARTITION_TO_SPLIT = {"train": "train", "dev": "valid", "test": "test"}
SLOT_RE = re.compile(r"\[([^\[\]:]+)\s*:\s*([^\[\]]+)\]")


def _parse_csv_arg(value: Optional[str]) -> Optional[set]:
    """Parse a comma-separated filter argument."""
    if not value:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def _download(url: str, output_path: Path) -> Path:
    """Download a URL with a tqdm progress bar."""
    ensure_dir(output_path.parent)
    progress = tqdm(unit="B", unit_scale=True, desc="Downloading MASSIVE")

    def reporthook(blocks: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            progress.total = total_size
        downloaded = blocks * block_size
        progress.update(max(0, downloaded - progress.n))

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
    finally:
        progress.close()
    return output_path


def _candidate_source_path(args: argparse.Namespace) -> Path:
    """Resolve MASSIVE input from local source or download cache."""
    if args.source:
        return args.source
    archive_name = args.download_url.rstrip("/").split("/")[-1] or "amazon-massive-dataset.tar.gz"
    archive_path = args.cache_dir / archive_name
    if args.download or not archive_path.exists():
        _download(args.download_url, archive_path)
    return archive_path


def _locale_jsonl_from_dir(source: Path, locale: str) -> Path:
    """Find a locale JSONL file under a MASSIVE directory."""
    candidates = [
        source / "{}.jsonl".format(locale),
        source / "data" / "{}.jsonl".format(locale),
        source / "1.0" / "data" / "{}.jsonl".format(locale),
        source / "1.1" / "data" / "{}.jsonl".format(locale),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(source.glob("**/{}.jsonl".format(locale)))
    if matches:
        return matches[0]
    raise FileNotFoundError("Could not find {}.jsonl under {}".format(locale, source))


def _read_jsonl_handle(handle: Iterable[bytes]) -> List[Dict[str, Any]]:
    """Read JSONL rows from a binary handle."""
    rows = []
    for raw_line in handle:
        line = raw_line.decode("utf-8-sig") if isinstance(raw_line, bytes) else raw_line.lstrip("\ufeff")
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_massive_rows(source: Path, locale: str) -> List[Dict[str, Any]]:
    """Read MASSIVE rows from a locale JSONL, directory, or tar.gz archive."""
    source = Path(source)
    if source.is_dir():
        with _locale_jsonl_from_dir(source, locale).open("rb") as handle:
            return _read_jsonl_handle(handle)
    if source.suffix == ".jsonl":
        with source.open("rb") as handle:
            return _read_jsonl_handle(handle)
    if source.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(source, "r:gz") as archive:
            candidates = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith("/data/{}.jsonl".format(locale))
            ]
            if not candidates:
                candidates = [
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith("{}.jsonl".format(locale))
                ]
            if not candidates:
                raise FileNotFoundError("Could not find {}.jsonl in {}".format(locale, source))
            handle = archive.extractfile(candidates[0])
            if handle is None:
                raise FileNotFoundError("Could not read {} from {}".format(candidates[0].name, source))
            return _read_jsonl_handle(handle)
    raise ValueError("Unsupported MASSIVE source: {}".format(source))


def parse_annotated_slots(annot_utt: str) -> Dict[str, str]:
    """Parse MASSIVE annot_utt slot annotations into a dict.

    Repeated slot labels are kept by suffixing later keys as ``slot__2``,
    ``slot__3``, and so on.
    """
    slots: Dict[str, str] = {}
    counts: Counter = Counter()
    for match in SLOT_RE.finditer(annot_utt or ""):
        slot_name = match.group(1).strip()
        value = match.group(2).strip()
        counts[slot_name] += 1
        key = slot_name if counts[slot_name] == 1 else "{}__{}".format(slot_name, counts[slot_name])
        slots[key] = value
    return slots


def _normalize_text_key(text: str) -> str:
    """Normalize text for deduplication only."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _passes_filters(
    row: Dict[str, Any],
    split: str,
    slots: Dict[str, str],
    intents: Optional[set],
    scenarios: Optional[set],
    min_slots: int,
    max_words: Optional[int],
) -> bool:
    """Return whether one MASSIVE row passes importer filters."""
    if split not in {"train", "valid", "test"}:
        return False
    if intents and row.get("intent") not in intents:
        return False
    if scenarios and row.get("scenario") not in scenarios:
        return False
    if len(slots) < min_slots:
        return False
    if max_words is not None and len(str(row.get("utt", "")).split()) > max_words:
        return False
    return True


def _sample_rows(rows: List[Dict[str, Any]], max_count: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """Deterministically sample up to max_count rows."""
    if max_count is None or len(rows) <= max_count:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_count))
    return [rows[index] for index in indices]


def _derive_metadata(splits: Dict[str, List[Dict[str, Any]]], args: argparse.Namespace, source: Path) -> Dict[str, Any]:
    """Build metadata for an imported MASSIVE dataset."""
    candidate_values_by_slot: Dict[str, set] = defaultdict(set)
    intent_slot_names: Dict[str, set] = defaultdict(set)
    intent_counts: Counter = Counter()
    scenario_counts: Counter = Counter()
    for rows in splits.values():
        for row in rows:
            intent = row["intent"]
            intent_counts[intent] += 1
            scenario_counts[row.get("scenario", "unknown")] += 1
            for slot_name, value in row.get("slots", {}).items():
                base_slot = slot_name.split("__", 1)[0]
                candidate_values_by_slot[base_slot].add(value)
                intent_slot_names[intent].add(base_slot)
    return {
        "dataset": "Amazon MASSIVE",
        "source": str(source),
        "download_url": args.download_url,
        "locale": args.locale,
        "seed": args.seed,
        "dedupe_text": args.dedupe_text,
        "min_slots": args.min_slots,
        "max_words": args.max_words,
        "intents_filter": sorted(_parse_csv_arg(args.intents) or []),
        "scenarios_filter": sorted(_parse_csv_arg(args.scenarios) or []),
        "train_size": len(splits["train"]),
        "valid_size": len(splits["valid"]),
        "test_size": len(splits["test"]),
        "intent_counts": dict(sorted(intent_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "candidate_values_by_slot": {
            slot: sorted(values) for slot, values in sorted(candidate_values_by_slot.items())
        },
        "intent_slot_names": {
            intent: sorted(slots) for intent, slots in sorted(intent_slot_names.items())
        },
    }


def convert_massive_rows(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Convert MASSIVE rows to RQ1 train/valid/test records."""
    intents = _parse_csv_arg(args.intents)
    scenarios = _parse_csv_arg(args.scenarios)
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    seen_text = set()
    for row in rows:
        partition = row.get("partition")
        split = PARTITION_TO_SPLIT.get(partition)
        slots = parse_annotated_slots(row.get("annot_utt", ""))
        if not _passes_filters(row, split, slots, intents, scenarios, args.min_slots, args.max_words):
            continue
        text = str(row.get("utt", "")).strip()
        if not text:
            continue
        text_key = _normalize_text_key(text)
        if args.dedupe_text and text_key in seen_text:
            continue
        seen_text.add(text_key)
        splits[split].append(
            {
                "id": "massive_{}_{}".format(split, row.get("id")),
                "split": split,
                "text": text,
                "intent": row.get("intent", "unknown"),
                "slots": slots,
                "scenario": row.get("scenario", "unknown"),
                "source": "massive",
                "massive_id": row.get("id"),
            }
        )

    max_counts = {"train": args.max_train, "valid": args.max_valid, "test": args.max_test}
    for offset, split in enumerate(["train", "valid", "test"]):
        sampled = _sample_rows(splits[split], max_counts[split], args.seed + offset)
        for index, row in enumerate(sampled):
            row["id"] = "massive_{}_{:06d}".format(split, index)
        splits[split] = sampled
    return splits


def import_massive(args: argparse.Namespace) -> Dict[str, Path]:
    """Import MASSIVE into RQ1 JSONL files."""
    source = _candidate_source_path(args)
    massive_rows = read_massive_rows(source, args.locale)
    splits = convert_massive_rows(massive_rows, args)
    ensure_dir(args.output_dir)
    output_paths = {}
    for split, rows in splits.items():
        path = args.output_dir / "{}.jsonl".format(split)
        write_jsonl(path, rows)
        output_paths[split] = path
    all_rows = splits["train"] + splits["valid"] + splits["test"]
    all_path = args.output_dir / "all.jsonl"
    write_jsonl(all_path, all_rows)
    output_paths["all"] = all_path
    metadata = _derive_metadata(splits, args, source)
    metadata_path = args.output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    output_paths["metadata"] = metadata_path
    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MASSIVE import CLI parser."""
    parser = argparse.ArgumentParser(description="Import Amazon MASSIVE into the RQ1 JSONL schema.")
    parser.add_argument("--source", default=None, type=Path, help="Local MASSIVE JSONL, directory, or tar.gz archive.")
    parser.add_argument("--download", action="store_true", help="Download MASSIVE if --source is not provided.")
    parser.add_argument("--download-url", default=DEFAULT_MASSIVE_URL)
    parser.add_argument("--cache-dir", default="data/raw/massive", type=Path)
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--output-dir", default="data/rq1_massive", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--min-slots", default=0, type=int)
    parser.add_argument("--max-words", default=30, type=int)
    parser.add_argument("--intents", default=None, help="Optional comma-separated intent filter.")
    parser.add_argument("--scenarios", default=None, help="Optional comma-separated scenario filter.")
    parser.add_argument("--max-train", default=None, type=int)
    parser.add_argument("--max-valid", default=None, type=int)
    parser.add_argument("--max-test", default=None, type=int)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--dedupe-text", dest="dedupe_text", action="store_true")
    parser.add_argument("--no-dedupe-text", dest="dedupe_text", action="store_false")
    parser.set_defaults(dedupe_text=True)
    return parser


def main(args: argparse.Namespace) -> Dict[str, Path]:
    """Run the MASSIVE import step from parsed CLI args."""
    if args.quick_test:
        args.max_train = 200
        args.max_valid = 50
        args.max_test = 50
    return import_massive(args)
