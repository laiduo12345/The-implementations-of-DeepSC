"""Plot RQ1 metric summaries."""

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from deepsc_ext.rq1.common import ensure_dir


def _load_summary(path: Path) -> List[Dict[str, float]]:
    """Load rq1_summary.csv into numeric records."""
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            parsed["snr"] = float(row["snr"])
            for key in [
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "intent_accuracy",
                "slot_precision",
                "slot_recall",
                "slot_f1",
                "task_success_rate",
            ]:
                parsed[key] = float(row[key])
            rows.append(parsed)  # type: ignore[arg-type]
    if not rows:
        raise ValueError("No rows found in {}".format(path))
    return rows  # type: ignore[return-value]


def _methods(rows: Sequence[Dict[str, float]]) -> List[str]:
    """Return methods in stable display order."""
    found = sorted({str(row["method"]) for row in rows})
    preferred = [method for method in ["full", "no_mi"] if method in found]
    return preferred + [method for method in found if method not in preferred]


def _series(rows: Sequence[Dict[str, float]], method: str, metric: str) -> List[Dict[str, float]]:
    """Return sorted rows for one method and metric."""
    values = [row for row in rows if str(row["method"]) == method and metric in row]
    values.sort(key=lambda row: float(row["snr"]))
    if not values:
        warnings.warn("No rows for method={} metric={}".format(method, metric))
    return values


def _save_current(figures_dir: Path, filename: str, save_pdf: bool) -> None:
    """Save the current matplotlib figure."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    png_path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    if save_pdf:
        plt.savefig(png_path.with_suffix(".pdf"))
    plt.close()
    print("Wrote {}".format(png_path))


def _plot_metric(
    rows: Sequence[Dict[str, float]],
    figures_dir: Path,
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
    save_pdf: bool,
) -> None:
    """Plot one metric across SNR for each method."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    plt.figure(figsize=(7, 4.5))
    for method in _methods(rows):
        values = _series(rows, method, metric)
        if not values:
            continue
        plt.plot(
            [row["snr"] for row in values],
            [row[metric] for row in values],
            marker="o",
            linewidth=2,
            label=method,
        )
    plt.title(title)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    _save_current(figures_dir, filename, save_pdf)


def _plot_bleu(rows: Sequence[Dict[str, float]], figures_dir: Path, save_pdf: bool) -> None:
    """Plot BLEU-1 and BLEU-4 across SNR for every method."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    plt.figure(figsize=(7.5, 4.8))
    for method in _methods(rows):
        for metric, style in [("bleu_1", "-"), ("bleu_4", "--")]:
            values = _series(rows, method, metric)
            if not values:
                continue
            plt.plot(
                [row["snr"] for row in values],
                [row[metric] for row in values],
                linestyle=style,
                marker="o",
                linewidth=2,
                label="{} {}".format(method, metric.replace("_", "-").upper()),
            )
    plt.title("RQ1 BLEU vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLEU")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    _save_current(figures_dir, "bleu_vs_snr.png", save_pdf)


def _plot_combined(rows: Sequence[Dict[str, float]], figures_dir: Path, save_pdf: bool) -> None:
    """Plot intent accuracy, slot F1, and task success together."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    metric_labels = [
        ("intent_accuracy", "Intent Acc"),
        ("slot_f1", "Slot F1"),
        ("task_success_rate", "Task Success"),
    ]
    plt.figure(figsize=(8, 5))
    for method in _methods(rows):
        for metric, label in metric_labels:
            values = _series(rows, method, metric)
            if not values:
                continue
            plt.plot(
                [row["snr"] for row in values],
                [row[metric] for row in values],
                marker="o",
                linewidth=2,
                label="{} {}".format(method, label),
            )
    plt.title("RQ1 Task Metrics vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    _save_current(figures_dir, "rq1_combined_metrics_vs_snr.png", save_pdf)


def plot_results(summary_csv: Path, output_dir: Path, save_pdf: bool = False) -> List[Path]:
    """Create RQ1 PNG figures from the summary CSV."""
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
    except ImportError as exc:
        raise ImportError("matplotlib is required for RQ1 plotting. Install project dependencies first.") from exc

    ensure_dir(output_dir)
    rows = _load_summary(summary_csv)
    _plot_bleu(rows, output_dir, save_pdf)
    _plot_metric(
        rows,
        output_dir,
        "intent_accuracy",
        "intent_accuracy_vs_snr.png",
        "RQ1 Intent Accuracy vs SNR",
        "Intent Accuracy",
        save_pdf,
    )
    _plot_metric(rows, output_dir, "slot_f1", "slot_f1_vs_snr.png", "RQ1 Slot F1 vs SNR", "Slot F1", save_pdf)
    _plot_metric(
        rows,
        output_dir,
        "task_success_rate",
        "task_success_vs_snr.png",
        "RQ1 Task Success Rate vs SNR",
        "Task Success Rate",
        save_pdf,
    )
    _plot_combined(rows, output_dir, save_pdf)
    return sorted(output_dir.glob("*.png"))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ1 plotting CLI parser."""
    parser = argparse.ArgumentParser(description="Plot RQ1 metrics from rq1_summary.csv.")
    parser.add_argument("--summary-csv", default="outputs/rq1_task_semantics/metrics/rq1_summary.csv", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq1_task_semantics/figures", type=Path)
    parser.add_argument("--save-pdf", action="store_true")
    return parser


def main(args: argparse.Namespace) -> List[Path]:
    """Run plotting from parsed CLI args."""
    return plot_results(args.summary_csv, args.output_dir, save_pdf=args.save_pdf)
