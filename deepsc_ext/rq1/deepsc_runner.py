"""Training helpers for the RQ1 DeepSC experiments."""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from deepsc_ext.rq1.common import command_string, ensure_dir, read_json, set_global_seed, snr_to_noise, write_json
from models.transceiver import Mine, Transeiver
from utlis.trainer import eval_step, train_step


DEFAULT_MODEL_CONFIG = {
    "encoder_num_layer": 4,
    "encoder_d_model": 128,
    "encoder_d_ff": 512,
    "encoder_num_heads": 8,
    "encoder_dropout": 0.1,
    "decoder_num_layer": 4,
    "decoder_d_model": 128,
    "decoder_d_ff": 512,
    "decoder_num_heads": 8,
    "decoder_dropout": 0.1,
    "max_length": 35,
    "shuffle_size": 2000,
}


def _dataset_cardinality(dataset: tf.data.Dataset) -> Optional[int]:
    """Return dataset cardinality when TensorFlow can infer it."""
    try:
        card = tf.data.experimental.cardinality(dataset).numpy()
    except Exception:
        return None
    if card < 0:
        return None
    return int(card)


def load_vocab(data_dir: Path) -> Dict[str, Any]:
    """Load a DeepSC vocab from an RQ1 data directory."""
    return read_json(data_dir / "vocab.json")


def build_deepsc_args(
    data_dir: Path,
    batch_size: int,
    channel: str,
    train_snr: float = 6.0,
    test_snr: float = 6.0,
    shuffle_size: int = DEFAULT_MODEL_CONFIG["shuffle_size"],
    max_length: int = DEFAULT_MODEL_CONFIG["max_length"],
) -> SimpleNamespace:
    """Build the args object expected by the original DeepSC model code."""
    vocab = load_vocab(data_dir)
    token_to_idx = vocab["token_to_idx"]
    values: Dict[str, Any] = dict(DEFAULT_MODEL_CONFIG)
    values.update(
        {
            "bs": batch_size,
            "shuffle_size": shuffle_size,
            "channel": channel,
            "train_snr": train_snr,
            "test_snr": test_snr,
            "vocab_path": str(data_dir / "vocab.json"),
            "train_save_path": str(data_dir / "train_data.pkl"),
            "valid_save_path": str(data_dir / "valid_data.pkl"),
            "test_save_path": str(data_dir / "test_data.pkl"),
            "vocab_size": len(token_to_idx),
            "pad_idx": token_to_idx["<PAD>"],
            "start_idx": token_to_idx["<START>"],
            "end_idx": token_to_idx["<END>"],
            "max_length": max_length,
        }
    )
    return SimpleNamespace(**values)


def load_pickle_sequences(path: Path) -> List[List[int]]:
    """Load encoded DeepSC sequences from a pickle file."""
    import pickle  # pylint: disable=import-outside-toplevel

    with path.open("rb") as handle:
        return pickle.load(handle)


def build_dataset(
    path: Path,
    batch_size: int,
    shuffle: bool,
    shuffle_size: int,
    seed: int,
) -> tf.data.Dataset:
    """Build a deterministic DeepSC dataset from an encoded pickle file."""
    raw_data = load_pickle_sequences(path)
    data_input = tf.keras.preprocessing.sequence.pad_sequences(raw_data, padding="post")
    dataset = tf.data.Dataset.from_tensor_slices((data_input, data_input))
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def _average_loss(losses: Iterable[float]) -> float:
    """Average a loss sequence without division-by-zero errors."""
    losses = list(losses)
    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))


def _resolve_checkpoint(path: Optional[Path]) -> Optional[str]:
    """Resolve a checkpoint file or directory to a TensorFlow checkpoint prefix."""
    if not path:
        return None
    path = Path(path)
    if path.is_dir():
        return tf.train.latest_checkpoint(str(path))
    return str(path)


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse values to JSON-serializable values."""
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def train_deepsc(args: argparse.Namespace, argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Train or fine-tune DeepSC for one RQ1 method."""
    set_global_seed(args.seed)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.log_dir)

    model_args = build_deepsc_args(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        channel=args.channel,
        train_snr=args.train_snr,
        shuffle_size=args.shuffle_size,
        max_length=args.max_length,
    )
    train_dataset = build_dataset(
        args.data_dir / "train_data.pkl",
        args.batch_size,
        shuffle=True,
        shuffle_size=args.shuffle_size,
        seed=args.seed,
    )
    valid_dataset = build_dataset(
        args.data_dir / "valid_data.pkl",
        args.batch_size,
        shuffle=False,
        shuffle_size=args.shuffle_size,
        seed=args.seed,
    )

    net = Transeiver(model_args)
    mine_net = Mine()
    optim_net = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-8,
    )
    optim_mi = tf.keras.optimizers.Adam(learning_rate=args.mine_learning_rate)
    checkpoint = tf.train.Checkpoint(Transceiver=net)
    manager = tf.train.CheckpointManager(checkpoint, directory=str(args.checkpoint_dir), max_to_keep=3)

    restore_path = _resolve_checkpoint(args.resume_checkpoint)
    if restore_path:
        checkpoint.restore(restore_path).expect_partial()

    train_batches = _dataset_cardinality(train_dataset)
    valid_batches = _dataset_cardinality(valid_dataset)
    log_payload = {
        "command": command_string(argv or sys.argv),
        "args": _jsonable_args(args),
        "data_metadata": read_json(args.data_dir / "metadata.json")
        if (args.data_dir / "metadata.json").exists()
        else None,
        "train_batches": train_batches,
        "valid_batches": valid_batches,
        "resume_checkpoint": restore_path,
    }
    write_json(args.log_dir / "train_config.json", log_payload)

    n_std = snr_to_noise(args.train_snr)
    best_valid_loss = float("inf")
    best_checkpoint = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_mi_losses = []
        for inp, tar in train_dataset:
            train_loss, train_loss_mine, _ = train_step(
                inp,
                tar,
                net,
                mine_net,
                optim_net,
                optim_mi,
                args.channel,
                n_std,
                train_with_mine=args.train_with_mine,
            )
            train_losses.append(float(train_loss.numpy()))
            train_mi_losses.append(float(train_loss_mine.numpy()))

        valid_losses = []
        for inp, tar in valid_dataset:
            valid_loss = eval_step(inp, tar, net, args.channel, n_std)
            valid_losses.append(float(valid_loss.numpy()))

        row = {
            "epoch": epoch,
            "train_loss": _average_loss(train_losses),
            "train_mine_loss": _average_loss(train_mi_losses),
            "valid_loss": _average_loss(valid_losses),
        }
        history.append(row)
        if row["valid_loss"] < best_valid_loss:
            best_valid_loss = row["valid_loss"]
            best_checkpoint = manager.save(checkpoint_number=epoch)
        print(
            "Epoch {}/{} train_loss={:.4f} valid_loss={:.4f}".format(
                epoch,
                args.epochs,
                row["train_loss"],
                row["valid_loss"],
            )
        )

    final_checkpoint = manager.save(checkpoint_number=args.epochs)
    history_path = args.log_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_mine_loss", "valid_loss"])
        writer.writeheader()
        writer.writerows(history)

    result = {
        "method": "full" if args.train_with_mine else "no_mi",
        "best_valid_loss": best_valid_loss,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "history_path": str(history_path),
    }
    write_json(args.log_dir / "train_result.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ1 training CLI parser."""
    parser = argparse.ArgumentParser(description="Train DeepSC for the RQ1 task semantics experiment.")
    parser.add_argument("--data-dir", default="data/rq1_task_semantics/deepsc_format", type=Path)
    parser.add_argument("--checkpoint-dir", default="outputs/rq1_task_semantics/checkpoints/full", type=Path)
    parser.add_argument("--log-dir", default=None, type=Path)
    parser.add_argument("--channel", default="AWGN", type=str)
    parser.add_argument("--train-snr", default=6.0, type=float)
    parser.add_argument("--batch-size", "--bs", dest="batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--learning-rate", default=5e-4, type=float)
    parser.add_argument("--mine-learning-rate", default=1e-3, type=float)
    parser.add_argument("--shuffle-size", default=DEFAULT_MODEL_CONFIG["shuffle_size"], type=int)
    parser.add_argument("--max-length", default=DEFAULT_MODEL_CONFIG["max_length"], type=int)
    parser.add_argument("--resume-checkpoint", default=None, type=Path)
    mine_group = parser.add_mutually_exclusive_group()
    mine_group.add_argument("--train-with-mine", dest="train_with_mine", action="store_true")
    mine_group.add_argument("--no-train-with-mine", dest="train_with_mine", action="store_false")
    parser.set_defaults(train_with_mine=True)
    parser.add_argument("--quick-test", action="store_true")
    return parser


def main(args: argparse.Namespace, argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run RQ1 DeepSC training from parsed CLI args."""
    if args.quick_test:
        args.epochs = 1
    if args.log_dir is None:
        args.log_dir = args.checkpoint_dir / "logs"
    return train_deepsc(args, argv=argv)
