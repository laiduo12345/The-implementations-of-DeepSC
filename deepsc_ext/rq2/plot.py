"""Plot RQ2 symbol efficiency results."""

import argparse
import csv
import math
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

from deepsc_ext.rq1.common import ensure_dir, format_snr


METRIC_LABELS = {
    "task_success_rate": "Task Success Rate",
    "intent_accuracy": "Intent Accuracy",
    "slot_f1": "Slot F1",
    "bleu_4": "BLEU-4",
}


def _load_summary(path: Path) -> List[Dict[str, object]]:
    """Load RQ2 summary CSV into numeric records."""
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, object] = dict(row)
            parsed["method"] = str(row["method"])
            parsed["symbols_per_word"] = int(row["symbols_per_word"])
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
                "avg_sentence_len",
                "avg_channel_uses",
                "task_success_per_symbol",
            ]:
                parsed[key] = float(row[key])
            rows.append(parsed)
    if not rows:
        raise ValueError("No rows found in {}".format(path))
    return rows


def _load_thresholds(path: Path) -> List[Dict[str, object]]:
    """Load RQ2 threshold CSV rows."""
    if not path.exists():
        warnings.warn("Threshold CSV '{}' does not exist; skipping threshold plot.".format(path))
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, object] = dict(row)
            parsed["method"] = str(row["method"])
            parsed["snr"] = float(row["snr"])
            parsed["threshold"] = float(row["threshold"])
            parsed["min_symbols_per_word"] = (
                float(row["min_symbols_per_word"]) if row.get("min_symbols_per_word") else math.nan
            )
            rows.append(parsed)
    return rows


def _methods(rows: Sequence[Dict[str, object]]) -> List[str]:
    """Return methods in stable display order."""
    found = sorted({str(row["method"]) for row in rows})
    preferred = [method for method in ["full", "no_mi"] if method in found]
    return preferred + [method for method in found if method not in preferred]


def _save(figures_dir: Path, filename: str, save_pdf: bool) -> None:
    """Save the current matplotlib figure."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    ensure_dir(figures_dir)
    path = figures_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    if save_pdf:
        plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    print("Wrote {}".format(path))


def _plot_metric_vs_symbols(
    rows: Sequence[Dict[str, object]],
    figures_dir: Path,
    metric: str,
    filename: str,
    save_pdf: bool,
) -> None:
    """Plot one metric against symbols per word, faceted by method."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    methods = _methods(rows)
    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 4.6), squeeze=False)
    for axis, method in zip(axes[0], methods):
        method_rows = [row for row in rows if row["method"] == method]
        snrs = sorted({float(row["snr"]) for row in method_rows})
        for snr in snrs:
            values = [row for row in method_rows if float(row["snr"]) == snr]
            values.sort(key=lambda row: int(row["symbols_per_word"]))
            axis.plot(
                [int(row["symbols_per_word"]) for row in values],
                [float(row[metric]) for row in values],
                marker="o",
                linewidth=2,
                label="SNR {} dB".format(format_snr(snr)),
            )
        axis.set_title("{} ({})".format(METRIC_LABELS[metric], method))
        axis.set_xlabel("Symbols per Word")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_ylim(0, 1.05)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(fontsize=8)
    _save(figures_dir, filename, save_pdf)


def _plot_heatmap(
    rows: Sequence[Dict[str, object]],
    figures_dir: Path,
    method: str,
    metric: str,
    filename: str,
    save_pdf: bool,
) -> None:
    """Plot one method's symbols x SNR heatmap for a metric."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        warnings.warn("No rows for method {}; skipping {}".format(method, filename))
        return
    symbols = sorted({int(row["symbols_per_word"]) for row in method_rows})
    snrs = sorted({float(row["snr"]) for row in method_rows})
    matrix = np.full((len(snrs), len(symbols)), np.nan)
    for row in method_rows:
        y = snrs.index(float(row["snr"]))
        x = symbols.index(int(row["symbols_per_word"]))
        matrix[y, x] = float(row[metric])

    plt.figure(figsize=(7.2, 4.8))
    axis = plt.gca()
    image = axis.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
    axis.set_title("RQ2 {} Heatmap ({})".format(METRIC_LABELS[metric], method))
    axis.set_xlabel("Symbols per Word")
    axis.set_ylabel("SNR (dB)")
    axis.set_xticks(range(len(symbols)))
    axis.set_xticklabels(symbols)
    axis.set_yticks(range(len(snrs)))
    axis.set_yticklabels([format_snr(snr) for snr in snrs])
    for y_index, snr in enumerate(snrs):
        for x_index, symbol in enumerate(symbols):
            value = matrix[y_index, x_index]
            if not np.isnan(value):
                axis.text(x_index, y_index, "{:.2f}".format(value), ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=axis, label=METRIC_LABELS[metric])
    _save(figures_dir, filename, save_pdf)


def _plot_minimal_symbols(
    threshold_rows: Sequence[Dict[str, object]],
    figures_dir: Path,
    save_pdf: bool,
) -> None:
    """Plot minimal symbols per word versus SNR for each threshold."""
    if not threshold_rows:
        return
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    methods = _methods(threshold_rows)
    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 4.6), squeeze=False)
    for axis, method in zip(axes[0], methods):
        method_rows = [row for row in threshold_rows if row["method"] == method]
        thresholds = sorted({float(row["threshold"]) for row in method_rows})
        for threshold in thresholds:
            values = [row for row in method_rows if float(row["threshold"]) == threshold]
            values.sort(key=lambda row: float(row["snr"]))
            axis.plot(
                [float(row["snr"]) for row in values],
                [float(row["min_symbols_per_word"]) for row in values],
                marker="o",
                linewidth=2,
                label="TSR >= {:.2f}".format(threshold),
            )
        axis.set_title("Minimal Symbols ({})".format(method))
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("Minimal Symbols per Word")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(fontsize=8)
    _save(figures_dir, "minimal_symbols_vs_snr.png", save_pdf)


def plot_results(summary_csv: Path, thresholds_csv: Path, output_dir: Path, save_pdf: bool = False) -> List[Path]:
    """Create RQ2 report figures from summary and threshold CSVs."""
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg")
    except ImportError as exc:
        raise ImportError("matplotlib is required for RQ2 plotting. Install project dependencies first.") from exc

    ensure_dir(output_dir)
    rows = _load_summary(summary_csv)
    _plot_metric_vs_symbols(rows, output_dir, "task_success_rate", "task_success_vs_symbols.png", save_pdf)
    _plot_metric_vs_symbols(rows, output_dir, "intent_accuracy", "intent_accuracy_vs_symbols.png", save_pdf)
    _plot_metric_vs_symbols(rows, output_dir, "slot_f1", "slot_f1_vs_symbols.png", save_pdf)
    _plot_metric_vs_symbols(rows, output_dir, "bleu_4", "bleu4_vs_symbols.png", save_pdf)

    for method in ["full", "no_mi"]:
        _plot_heatmap(
            rows,
            output_dir,
            method,
            "task_success_rate",
            "task_success_heatmap_{}.png".format(method),
            save_pdf,
        )
        _plot_heatmap(rows, output_dir, method, "slot_f1", "slot_f1_heatmap_{}.png".format(method), save_pdf)

    threshold_rows = _load_thresholds(thresholds_csv)
    _plot_minimal_symbols(threshold_rows, output_dir, save_pdf)
    return sorted(output_dir.glob("*.png"))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RQ2 plotting CLI parser."""
    parser = argparse.ArgumentParser(description="Plot RQ2 symbol efficiency results.")
    parser.add_argument("--summary-csv", default="outputs/rq2_symbol_efficiency/metrics/rq2_summary.csv", type=Path)
    parser.add_argument("--thresholds-csv", default="outputs/rq2_symbol_efficiency/metrics/rq2_thresholds.csv", type=Path)
    parser.add_argument("--output-dir", default="outputs/rq2_symbol_efficiency/figures", type=Path)
    parser.add_argument("--save-pdf", action="store_true")
    return parser


def main(args: argparse.Namespace) -> List[Path]:
    """Run RQ2 plotting from parsed CLI args."""
    return plot_results(args.summary_csv, args.thresholds_csv, args.output_dir, save_pdf=args.save_pdf)

