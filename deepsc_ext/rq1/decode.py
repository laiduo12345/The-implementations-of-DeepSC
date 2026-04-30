"""Decode RQ1 test samples across SNR values."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from deepsc_ext.rq1.common import ensure_dir, format_snr, parse_snrs, read_json, read_jsonl, set_global_seed, snr_to_noise, write_jsonl
from deepsc_ext.rq1.deepsc_runner import build_deepsc_args, load_pickle_sequences
from models.transceiver import Transeiver
from utlis.trainer import greedy_decode


def _build_decode_dataset(path: Path, batch_size: int) -> tf.data.Dataset:
    """Build a non-shuffled decode dataset from encoded sequences."""
    raw_data = load_pickle_sequences(path)
    data_input = tf.keras.preprocessing.sequence.pad_sequences(raw_data, padding="post")
    return tf.data.Dataset.from_tensor_slices(data_input).batch(batch_size)


def _idx_to_token(vocab: Dict[str, Any]) -> Dict[int, str]:
    """Build an index-to-token map from a DeepSC vocab."""
    return {int(index): token for token, index in vocab["token_to_idx"].items()}


def sequence_to_clean_text(sequence: List[int], vocab: Dict[str, Any]) -> str:
    """Convert decoded token ids to text while removing DeepSC special tokens."""
    idx_to_token = _idx_to_token(vocab)
    token_to_idx = vocab["token_to_idx"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    pad_idx = token_to_idx["<PAD>"]
    tokens = []
    for raw_idx in sequence:
        idx = int(raw_idx)
        if idx == end_idx:
            break
        if idx in (start_idx, pad_idx):
            continue
        token = idx_to_token.get(idx)
        if token:
            tokens.append(token)
    return " ".join(tokens).strip()


def _resolve_latest_checkpoint(checkpoint_dir: Path) -> str:
    """Return the latest checkpoint prefix in a directory."""
    checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if not checkpoint:
        raise FileNotFoundError("No TensorFlow checkpoint found in {}".format(checkpoint_dir))
    return checkpoint


def decode_snr(args: argparse.Namespace) -> List[Path]:
    """Decode the RQ1 test set for each requested SNR and write JSONL outputs."""
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)
    test_rows = read_jsonl(args.test_jsonl)
    raw_sequences = load_pickle_sequences(args.data_dir / "test_data.pkl")
    if len(test_rows) != len(raw_sequences):
        raise ValueError(
            "test JSONL count ({}) does not match test_data.pkl count ({})".format(
                len(test_rows),
                len(raw_sequences),
            )
        )

    model_args = build_deepsc_args(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        channel=args.channel,
        test_snr=args.snrs[0] if args.snrs else 6.0,
        max_length=args.max_length,
        symbols_per_word=getattr(args, "symbols_per_word", 8),
    )
    vocab = read_json(args.data_dir / "vocab.json")
    net = Transeiver(model_args)
    checkpoint = tf.train.Checkpoint(Transceiver=net)
    checkpoint_path = _resolve_latest_checkpoint(args.checkpoint_dir)
    checkpoint.restore(checkpoint_path).expect_partial()
    dataset = _build_decode_dataset(args.data_dir / "test_data.pkl", args.batch_size)

    output_paths = []
    for snr in args.snrs:
        set_global_seed(args.seed + int(float(snr) * 1000))
        decoded_texts: List[str] = []
        n_std = snr_to_noise(snr)
        for batch in dataset:
            preds = greedy_decode(model_args, batch, net, args.channel, n_std)
            decoded_texts.extend(sequence_to_clean_text(row, vocab) for row in preds.numpy().tolist())

        rows = []
        for source, decoded in zip(test_rows, decoded_texts):
            rows.append(
                {
                    "id": source["id"],
                    "snr": int(snr) if float(snr).is_integer() else snr,
                    "method": args.method,
                    "original_text": source["text"],
                    "decoded_text": decoded,
                    "intent": source["intent"],
                    "slots": source["slots"],
                }
            )
        output_path = args.output_dir / "snr_{}.jsonl".format(format_snr(snr))
        write_jsonl(output_path, rows)
        output_paths.append(output_path)
        print("Wrote {}".format(output_path))
    return output_paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ1 decode CLI parser."""
    parser = argparse.ArgumentParser(description="Decode RQ1 test samples across SNR values.")
    parser.add_argument("--data-dir", default="data/rq1_task_semantics/deepsc_format", type=Path)
    parser.add_argument("--test-jsonl", default="data/rq1_task_semantics/test.jsonl", type=Path)
    parser.add_argument("--checkpoint-dir", default="outputs/rq1_task_semantics/checkpoints/full", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq1_task_semantics/decoded/full", type=Path)
    parser.add_argument("--channel", default="AWGN", type=str)
    parser.add_argument("--snrs", default="0,3,6,9,12,15,18", type=str)
    parser.add_argument("--batch-size", "--bs", dest="batch_size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--method", default="full", type=str)
    parser.add_argument("--max-length", default=35, type=int)
    parser.add_argument("--symbols-per-word", default=8, type=int)
    return parser


def main(args: argparse.Namespace) -> List[Path]:
    """Run the decode step from parsed CLI args."""
    if isinstance(args.snrs, str):
        args.snrs = parse_snrs(args.snrs)
    return decode_snr(args)
