# RQ2: Task-Level Symbol Efficiency

RQ2 studies how task-level semantic quality changes when DeepSC uses different channel-symbol budgets. RQ1 fixes the symbols per word and sweeps SNR; RQ2 trains a separate model for each symbols-per-word value and evaluates the rate-performance tradeoff.

## Symbols Per Word

`symbols_per_word = N` means each input word is represented by `N` complex channel symbols. The model uses real-valued tensors, so the channel encoder output dimension is `2 * N`.

The original repository used a fixed output dimension of `16`, so the default remains `symbols_per_word = 8`. RQ2 supports values such as `1,2,3,4,6,8,10`.

## Data

RQ2 reuses the RQ1 MASSIVE DeepSC-format data by default:

```text
data/rq1_massive/
  train.jsonl
  valid.jsonl
  test.jsonl
  deepsc_format/
    train_data.pkl
    valid_data.pkl
    test_data.pkl
    vocab.json
    metadata.json
```

The decoded JSONL format extends RQ1 with `symbols_per_word`.

## Quick Test

```powershell
python scripts/rq2_run_pipeline.py `
  --stage all `
  --quick-test `
  --output-dir outputs/rq2_symbol_efficiency_quick
```

Quick-test defaults:

- `symbols-list = 1,4`
- `snrs = -9,0`
- `methods = no_mi`
- `epochs = 1`
- train/valid/test prefixes: `200/50/100`

## One-Command Experiments

Fixed SNR sweep:

```powershell
python scripts/rq2_run_pipeline.py `
  --stage all `
  --mode fixed_snr `
  --fixed-snr -3 `
  --symbols-list 1,2,3,4,6,8,10 `
  --methods full,no_mi `
  --output-dir outputs/rq2_symbol_efficiency
```

Full symbols x SNR grid:

```powershell
python scripts/rq2_run_pipeline.py   --stage all   --mode grid   --symbols-list 1,2,3,4,6,8,10   --snrs -15,-12,-9,-6,-3,0,3,6,9,12   --methods full,no_mi   --output-dir outputs/rq2_symbol_efficiency
```

## Staged Commands

Run these in order to match the full grid one-command above.

Train all checkpoints:

```powershell
$methods = "full","no_mi"
$symbols = 1,2,3,4,6,8,10

foreach ($method in $methods) {
  foreach ($spw in $symbols) {
    python scripts/rq2_train_deepsc.py `
      --data-dir data/rq1_massive/deepsc_format `
      --checkpoint-dir "outputs/rq2_symbol_efficiency/checkpoints/$method/spw_$spw" `
      --log-dir "outputs/rq2_symbol_efficiency/logs/train_${method}_spw_${spw}" `
      --method $method `
      --symbols-per-word $spw `
      --channel AWGN `
      --train-snr 6 `
      --batch-size 64 `
      --epochs 20 `
      --seed 42 `
      --max-length 35
  }
}
```

Decode a grid:

```powershell
python scripts/rq2_decode_grid.py `
  --data-dir data/rq1_massive/deepsc_format `
  --test-jsonl data/rq1_massive/test.jsonl `
  --checkpoint-root outputs/rq2_symbol_efficiency/checkpoints `
  --output-dir outputs/rq2_symbol_efficiency/decoded `
  --symbols-list 1,2,3,4,6,8,10 `
  --snrs -15,-12,-9,-6,-3,0,3,6,9,12 `
  --methods full,no_mi `
  --mode grid `
  --fixed-snr 0 `
  --channel AWGN `
  --batch-size 256 `
  --seed 42 `
  --max-length 35
```

Evaluate:

```powershell
python scripts/rq2_evaluate_metrics.py `
  --decoded-dir outputs/rq2_symbol_efficiency/decoded `
  --output-dir outputs/rq2_symbol_efficiency/metrics `
  --metadata-json data/rq1_massive/deepsc_format/metadata.json `
  --methods full,no_mi `
  --symbols-list 1,2,3,4,6,8,10 `
  --snrs -15,-12,-9,-6,-3,0,3,6,9,12 `
  --fixed-snr 0
```

Thresholds:

```powershell
python scripts/rq2_summarize_thresholds.py `
  --summary-csv outputs/rq2_symbol_efficiency/metrics/rq2_summary.csv `
  --output-csv outputs/rq2_symbol_efficiency/metrics/rq2_thresholds.csv `
  --thresholds 0.8,0.9,0.95
```

Plot:

```powershell
python scripts/rq2_plot_results.py `
  --summary-csv outputs/rq2_symbol_efficiency/metrics/rq2_summary.csv `
  --thresholds-csv outputs/rq2_symbol_efficiency/metrics/rq2_thresholds.csv `
  --output-dir outputs/rq2_symbol_efficiency/figures
```

Pipeline stages:

```powershell
python scripts/rq2_run_pipeline.py --stage train ...
python scripts/rq2_run_pipeline.py --stage decode ...
python scripts/rq2_run_pipeline.py --stage evaluate ...
python scripts/rq2_run_pipeline.py --stage thresholds ...
python scripts/rq2_run_pipeline.py --stage plot ...
```

## Outputs

```text
outputs/rq2_symbol_efficiency/
  configs/rq2_config.json
  checkpoints/{method}/spw_{N}/
    config.json
    train.log
  decoded/{method}/spw_{N}/snr_{snr}.jsonl
  metrics/
    rq2_summary.csv
    rq2_cases.csv
    rq2_thresholds.csv
    rq2_fixed_snr_summary.csv
  figures/
    task_success_vs_symbols.png
    intent_accuracy_vs_symbols.png
    slot_f1_vs_symbols.png
    bleu4_vs_symbols.png
    task_success_heatmap_full.png
    task_success_heatmap_no_mi.png
    slot_f1_heatmap_full.png
    slot_f1_heatmap_no_mi.png
    minimal_symbols_vs_snr.png
  logs/
```

## CSV Fields

`rq2_summary.csv` contains:

```text
method,symbols_per_word,snr,num_samples,bleu_1,bleu_2,bleu_3,bleu_4,
intent_accuracy,slot_precision,slot_recall,slot_f1,task_success_rate,
avg_sentence_len,avg_channel_uses,task_success_per_symbol
```

`avg_channel_uses = avg_sentence_len * symbols_per_word`.

`rq2_cases.csv` contains sample-level decoded text, predicted task metrics, `sentence_len`, and `channel_uses`.

`rq2_thresholds.csv` contains the minimal symbols per word needed to reach each Task Success Rate threshold for each method and SNR.

## Figures

- `*_vs_symbols.png`: metric trends as the symbol budget changes.
- `task_success_heatmap_*.png`: Task Success Rate over symbols x SNR.
- `slot_f1_heatmap_*.png`: Slot F1 over symbols x SNR.
- `minimal_symbols_vs_snr.png`: minimal symbol budget needed for thresholds such as `0.8`, `0.9`, and `0.95`.

## Known Issues

- Intent and slot metrics reuse the RQ1 rule-based evaluator; they are task-semantics proxies, not outputs from a separately trained NLU model.
- Quick-test is only a pipeline check and should not be reported as final performance.
- Each `symbols_per_word` value must be trained separately; changing it only at decode time is intentionally rejected.
- The current pipeline targets AWGN unless another channel is explicitly passed.

