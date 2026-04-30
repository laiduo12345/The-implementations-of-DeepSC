"""Grid decoding for RQ2 symbol efficiency experiments."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from deepsc_ext.rq1.common import ensure_dir, read_json, read_jsonl, set_global_seed, snr_to_noise, write_jsonl
from deepsc_ext.rq1.decode import sequence_to_clean_text
from deepsc_ext.rq1.deepsc_runner import build_deepsc_args, load_pickle_sequences
from deepsc_ext.rq2.common import (
    DEFAULT_METHODS,
    DEFAULT_SNRS,
    DEFAULT_SYMBOLS,
    active_snrs,
    list_to_csv,
    maybe_int,
    parse_float_list,
    parse_int_list,
    parse_methods,
    snr_filename,
    spw_dir_name,
    validate_checkpoint_symbols,
)
from models.transceiver import Transeiver
from utlis.trainer import greedy_decode


def _build_decode_dataset(path: Path, batch_size: int, max_samples: int = None) -> tf.data.Dataset:
    """Build a non-shuffled decode dataset from encoded sequences."""
    raw_data = load_pickle_sequences(path)
    if max_samples is not None and max_samples > 0:
        raw_data = raw_data[:max_samples]
    data_input = tf.keras.preprocessing.sequence.pad_sequences(raw_data, padding="post")
    return tf.data.Dataset.from_tensor_slices(data_input).batch(batch_size)


def _limited_rows(rows: List[Dict[str, Any]], max_samples: int = None) -> List[Dict[str, Any]]:
    """Return a stable prefix when a sample limit is requested."""
    if max_samples is not None and max_samples > 0:
        return rows[:max_samples]
    return rows


def _decode_one(
    args: argparse.Namespace,
    method: str,
    symbols_per_word: int,
    snrs: Sequence[float],
    test_rows: List[Dict[str, Any]],
) -> List[Path]:
    """Decode one method/symbols checkpoint across SNRs."""
    checkpoint_dir = args.checkpoint_root / method / spw_dir_name(symbols_per_word)
    checkpoint_path = validate_checkpoint_symbols(
        checkpoint_dir,
        symbols_per_word,
        strict=args.strict_checkpoints,
    )
    if checkpoint_path is None:
        return []

    model_args = build_deepsc_args(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        channel=args.channel,
        test_snr=snrs[0] if snrs else 6.0,
        max_length=args.max_length,
        symbols_per_word=symbols_per_word,
    )
    vocab = read_json(args.data_dir / "vocab.json")
    net = Transeiver(model_args)
    checkpoint = tf.train.Checkpoint(Transceiver=net)
    checkpoint.restore(checkpoint_path).expect_partial()
    dataset = _build_decode_dataset(args.data_dir / "test_data.pkl", args.batch_size, args.max_test_samples)

    output_paths: List[Path] = []
    for snr in snrs:
        seed_offset = int(symbols_per_word * 10000 + float(snr) * 1000)
        set_global_seed(args.seed + seed_offset)
        n_std = snr_to_noise(snr)
        decoded_texts: List[str] = []
        for batch in dataset:
            preds = greedy_decode(model_args, batch, net, args.channel, n_std)
            decoded_texts.extend(sequence_to_clean_text(row, vocab) for row in preds.numpy().tolist())

        rows = []
        for source, decoded in zip(test_rows, decoded_texts):
            rows.append(
                {
                    "id": source.get("id", ""),
                    "method": method,
                    "symbols_per_word": int(symbols_per_word),
                    "snr": maybe_int(float(snr)),
                    "original_text": source.get("text", source.get("original_text", "")),
                    "decoded_text": decoded or "",
                    "intent": source.get("intent", ""),
                    "slots": source.get("slots", {}),
                    "scenario": source.get("scenario", ""),
                }
            )
        output_path = args.output_dir / method / spw_dir_name(symbols_per_word) / snr_filename(float(snr))
        write_jsonl(output_path, rows)
        output_paths.append(output_path)
        print("Wrote {}".format(output_path))
    return output_paths


def decode_grid(args: argparse.Namespace) -> List[Path]:
    """Decode all requested RQ2 method/symbol/SNR combinations."""
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)
    test_rows_all = read_jsonl(args.test_jsonl)
    raw_sequences = load_pickle_sequences(args.data_dir / "test_data.pkl")
    if len(test_rows_all) != len(raw_sequences):
        raise ValueError(
            "test JSONL count ({}) does not match test_data.pkl count ({})".format(
                len(test_rows_all),
                len(raw_sequences),
            )
        )
    test_rows = _limited_rows(test_rows_all, args.max_test_samples)
    snrs = active_snrs(args.mode, args.snrs, args.fixed_snr)

    output_paths: List[Path] = []
    for method in args.methods:
        for symbols_per_word in args.symbols_list:
            output_paths.extend(_decode_one(args, method, int(symbols_per_word), snrs, test_rows))
    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ2 grid decode CLI parser."""
    parser = argparse.ArgumentParser(description="Decode RQ2 checkpoints across symbols-per-word and SNR values.")
    parser.add_argument("--data-dir", default="data/rq1_massive/deepsc_format", type=Path)
    parser.add_argument("--test-jsonl", default="data/rq1_massive/test.jsonl", type=Path)
    parser.add_argument("--checkpoint-root", default="outputs/rq2_symbol_efficiency/checkpoints", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq2_symbol_efficiency/decoded", type=Path)
    parser.add_argument("--symbols-list", default=list_to_csv(DEFAULT_SYMBOLS))
    parser.add_argument("--snrs", default=list_to_csv(DEFAULT_SNRS))
    parser.add_argument("--methods", default=list_to_csv(DEFAULT_METHODS))
    parser.add_argument("--mode", default="grid", choices=["grid", "fixed_snr"])
    parser.add_argument("--fixed-snr", default=0.0, type=float)
    parser.add_argument("--channel", default="AWGN")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-length", default=35, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    parser.add_argument("--strict-checkpoints", action="store_true")
    return parser


def main(args: argparse.Namespace) -> List[Path]:
    """Run RQ2 grid decoding from parsed CLI args."""
    if isinstance(args.symbols_list, str):
        args.symbols_list = parse_int_list(args.symbols_list)
    if isinstance(args.snrs, str):
        args.snrs = parse_float_list(args.snrs)
    if isinstance(args.methods, str):
        args.methods = parse_methods(args.methods)
    return decode_grid(args)

