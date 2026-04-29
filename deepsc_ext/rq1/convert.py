"""Convert RQ1 JSONL task data into the DeepSC pickle/vocab format."""

import argparse
import json
import pickle
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from dataset.preprocess_text import SPECIAL_TOKENS, build_vocab, normalize_string, tokenize
from deepsc_ext.rq1.common import ensure_dir, read_jsonl, write_json
from deepsc_ext.rq1.data import candidate_values_by_slot


PUNCT_TO_KEEP = [";", ","]
PUNCT_TO_REMOVE = ["?", "."]


def normalize_for_deepsc(text: str) -> str:
    """Apply the original DeepSC text normalization."""
    return normalize_string(text).strip()


def encode_texts(texts: Iterable[str], token_to_idx: Dict[str, int]) -> List[List[int]]:
    """Encode normalized texts with the original DeepSC tokenizer."""
    encoded = []
    for text in texts:
        tokens = tokenize(
            text,
            punct_to_keep=PUNCT_TO_KEEP,
            punct_to_remove=PUNCT_TO_REMOVE,
        )
        encoded.append([token_to_idx[token] for token in tokens])
    return encoded


def _write_pickle(path: Path, value: Any) -> None:
    """Write a pickle file."""
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _length_summary(encoded_splits: Dict[str, List[List[int]]]) -> Dict[str, float]:
    """Compute length metadata over every encoded split."""
    lengths = [len(row) for rows in encoded_splits.values() for row in rows]
    return {
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": float(mean(lengths)) if lengths else 0.0,
    }


def convert_jsonl_to_deepsc(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Convert RQ1 JSONL splits to DeepSC-compatible pkl and vocab files."""
    ensure_dir(output_dir)
    split_paths = {split: input_dir / "{}.jsonl".format(split) for split in ["train", "valid", "test"]}
    rows = {split: read_jsonl(path) for split, path in split_paths.items()}
    normalized = {
        split: [normalize_for_deepsc(row["text"]) for row in split_rows]
        for split, split_rows in rows.items()
    }
    all_sentences = normalized["train"] + normalized["valid"] + normalized["test"]
    token_to_idx = build_vocab(
        all_sentences,
        dict(SPECIAL_TOKENS),
        punct_to_keep=PUNCT_TO_KEEP,
        punct_to_remove=PUNCT_TO_REMOVE,
    )
    encoded = {split: encode_texts(texts, token_to_idx) for split, texts in normalized.items()}

    for split, values in encoded.items():
        _write_pickle(output_dir / "{}_data.pkl".format(split), values)

    vocab = {"token_to_idx": token_to_idx}
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    lengths = _length_summary(encoded)
    metadata: Dict[str, Any] = {
        "train_size": len(rows["train"]),
        "valid_size": len(rows["valid"]),
        "test_size": len(rows["test"]),
        "vocab_size": len(token_to_idx),
        "seed": seed,
        "source_jsonl_paths": {split: str(path) for split, path in split_paths.items()},
        "candidate_values_by_slot": candidate_values_by_slot(),
    }
    metadata.update(lengths)
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the conversion CLI parser."""
    parser = argparse.ArgumentParser(description="Convert RQ1 JSONL data to DeepSC pickle format.")
    parser.add_argument("--input-dir", default="data/rq1_task_semantics", type=Path)
    parser.add_argument("--output-dir", default="data/rq1_task_semantics/deepsc_format", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the conversion step from parsed CLI args."""
    return convert_jsonl_to_deepsc(args.input_dir, args.output_dir, args.seed)
