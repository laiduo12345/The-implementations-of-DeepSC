# RQ1: Task-Level Semantic Preservation under Different SNRs

This experiment tests whether DeepSC preserves task-level semantics under different SNR values while keeping the original model structure and channel symbol dimension unchanged.

The pipeline generates template task data, converts it to the repository's DeepSC pickle format, trains two model groups, decodes the test set across SNRs, computes task metrics, and plots report-ready figures.

## Runtime

Use the repository Python 3.8 environment or another environment compatible with TensorFlow 2.3.x. In this checkout, the working interpreter is:

```powershell
.venv\Scripts\python.exe
```

The global Python 3.13 environment is not compatible with this repository's legacy Keras stack.

## Data

Generated JSONL rows have this shape:

```json
{"id":"train_000001","split":"train","text":"Book a flight from Paris to Tokyo on Friday.","intent":"book_flight","slots":{"from_city":"Paris","to_city":"Tokyo","date":"Friday"}}
```

Supported intents:

- `book_flight`: `from_city`, `to_city`, `date`
- `check_weather`: `city`, `date`
- `set_alarm`: `time`
- `play_music`: `song`, `artist`

DeepSC format outputs are pickle lists of token id sequences plus `vocab.json` with the original `{"token_to_idx": ...}` format.

## Quick Test

Run the full small pipeline:

```powershell
.venv\Scripts\python.exe scripts\rq1_run_pipeline.py --stage all --quick-test
```

Quick test settings:

- train/valid/test sizes: `200/50/50`
- epochs: `1`
- SNRs: `0,6`

## Full Pipeline

```powershell
.venv\Scripts\python.exe scripts\rq1_run_pipeline.py `
  --stage all `
  --snrs 0,3,6,9,12,15,18 `
  --output-dir outputs/rq1_task_semantics `
  --seed 42
```

Use `--skip-train` when checkpoints already exist and only decode/evaluate/plot stages are needed.

## Staged Commands

Generate template data:

```powershell
.venv\Scripts\python.exe scripts\rq1_generate_data.py `
  --output-dir data/rq1_task_semantics `
  --train-size 8000 `
  --valid-size 1000 `
  --test-size 1000 `
  --seed 42
```

Convert to DeepSC format:

```powershell
.venv\Scripts\python.exe scripts\rq1_convert_data.py `
  --input-dir data/rq1_task_semantics `
  --output-dir data/rq1_task_semantics/deepsc_format `
  --seed 42
```

Train the full model with MI loss:

```powershell
.venv\Scripts\python.exe scripts\rq1_train_deepsc.py `
  --data-dir data/rq1_task_semantics/deepsc_format `
  --checkpoint-dir outputs/rq1_task_semantics/checkpoints/full `
  --log-dir outputs/rq1_task_semantics/logs/train_full `
  --channel AWGN `
  --train-snr 6 `
  --batch-size 64 `
  --epochs 60 `
  --seed 42 `
  --train-with-mine
```

Train the no-MI control:

```powershell
.venv\Scripts\python.exe scripts\rq1_train_deepsc.py `
  --data-dir data/rq1_task_semantics/deepsc_format `
  --checkpoint-dir outputs/rq1_task_semantics/checkpoints/no_mi `
  --log-dir outputs/rq1_task_semantics/logs/train_no_mi `
  --channel AWGN `
  --train-snr 6 `
  --batch-size 64 `
  --epochs 60 `
  --seed 42 `
  --no-train-with-mine
```

Decode one method across SNRs:

```powershell
.venv\Scripts\python.exe scripts\rq1_decode_snr.py `
  --data-dir data/rq1_task_semantics/deepsc_format `
  --test-jsonl data/rq1_task_semantics/test.jsonl `
  --checkpoint-dir outputs/rq1_task_semantics/checkpoints/full `
  --output-dir outputs/rq1_task_semantics/decoded/full `
  --channel AWGN `
  --snrs 0,3,6,9,12,15,18 `
  --batch-size 256 `
  --seed 42 `
  --method full
```

Evaluate decoded JSONL files:

```powershell
.venv\Scripts\python.exe scripts\rq1_evaluate_task_metrics.py `
  --decoded-dir outputs/rq1_task_semantics/decoded `
  --output-dir outputs/rq1_task_semantics/metrics
```

Plot figures:

```powershell
.venv\Scripts\python.exe scripts\rq1_plot_results.py `
  --summary-csv outputs/rq1_task_semantics/metrics/rq1_summary.csv `
  --output-dir outputs/rq1_task_semantics/figures
```

## Outputs

```text
data/rq1_task_semantics/
  train.jsonl
  valid.jsonl
  test.jsonl
  all.jsonl
  deepsc_format/
    train_data.pkl
    valid_data.pkl
    test_data.pkl
    vocab.json
    metadata.json

outputs/rq1_task_semantics/
  checkpoints/
    full/
    no_mi/
  decoded/
    full/snr_*.jsonl
    no_mi/snr_*.jsonl
  metrics/
    rq1_summary.csv
    rq1_cases.csv
  figures/
    bleu_vs_snr.png
    intent_accuracy_vs_snr.png
    slot_f1_vs_snr.png
    task_success_vs_snr.png
    rq1_combined_metrics_vs_snr.png
  configs/
    rq1_config.json
  logs/
    train_full/
    train_no_mi/
```

## Metrics

- BLEU-1 through BLEU-4 use a local corpus BLEU implementation.
- Intent Accuracy uses keyword and entity fallback rules.
- Slot Precision, Recall, and F1 use normalized string matching against intent-specific slot candidate values.
- Task Success is `pred_intent == true_intent` and all gold slot values preserved.

## Known Limitations

- Data is synthetic template data.
- Intent classification is rule-based.
- Slot matching is string-level and does not handle paraphrases such as `seven am` vs `7 am`.
- Symbols per word and model architecture are unchanged.
- The first version targets AWGN only.
- Long training is not run automatically by this document; use `--quick-test` for pipeline validation before full experiments.
