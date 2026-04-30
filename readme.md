# DeepSC Experimental Extension

This repository is based on the original implementation of **Deep Learning Enabled Semantic Communication Systems** by Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang.

The original DeepSC training and evaluation entrypoints are still available. This fork adds reproducible experiment pipelines for task-level semantic communication research. The current extension focuses on two research questions:

- **RQ1: Task-Level Semantic Preservation under Different SNRs**  
  Evaluate whether decoded text preserves task intent and slot information under different channel SNR values.

- **RQ2: Task-Level Symbol Efficiency**  
  Evaluate how task-level semantic quality changes when the number of channel symbols per word is changed.

The main goal of these changes is to support a paper-oriented experimental workflow, not to replace the original DeepSC baseline.

## Repository Layout

```text
dataset/                         Original text preprocessing and data loading
models/                          Original DeepSC model and channel modules
utlis/                           Original training, decoding, and metric helpers
main.py                          Original DeepSC training entrypoint
evaluation.py                    Original DeepSC evaluation entrypoint

deepsc_ext/rq1/                  RQ1 data, conversion, training, decoding, metrics, plotting
deepsc_ext/rq2/                  RQ2 symbol-efficiency training, decoding, metrics, plotting
scripts/rq1_*.py                 RQ1 command-line scripts
scripts/rq2_*.py                 RQ2 command-line scripts
experiments/rq1_task_semantics/  RQ1 experiment guide
experiments/rq2_symbol_efficiency/ RQ2 experiment guide
docs/                            Dataset and paper-supporting notes
```

Generated datasets, checkpoints, decoded cases, metrics, and figures are written under `data/` and `outputs/`. These directories are ignored by git.

## Runtime Requirements

Use Python 3.8 with the legacy TensorFlow/Keras stack used by the original project.

The project dependencies are recorded in `pyproject.toml` and `uv.lock`:

```text
tensorflow==2.3.4
keras==2.3.1
bert4keras==0.11.5
protobuf==3.20.3
numpy
w3lib
tqdm
nltk
scikit-learn
matplotlib
```

If `uv` is available, create or synchronize the environment with:

```powershell
uv sync
```

Then run commands with the project environment, for example:

```powershell
.venv\Scripts\python.exe scripts\rq1_run_pipeline.py --help
```

The global Python 3.13 environment is not compatible with this repository's legacy TensorFlow/Keras dependencies.

## Original DeepSC Workflow

The original Europarl text preprocessing, training, and evaluation commands remain supported.

### Preprocess Europarl

```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python dataset/preprocess_text.py
```

The original data format is:

- `train_data.pkl`: list of token-id sequences
- `test_data.pkl`: list of token-id sequences
- `vocab.json`: `{"token_to_idx": ...}`

### Train Original DeepSC

```shell
python main.py --bs=64 --train-snr=6 --channel=AWGN --train-with-mine --checkpoint-path=./checkpoint
```

Important options:

| Option | Meaning |
|---|---|
| `--train-snr` | Training SNR |
| `--train-with-mine` | Enable MINE-based mutual information loss |
| `--channel` | Channel type, for example `AWGN` |
| `--bs` | Batch size |
| `--lr` | Learning rate |
| `--checkpoint-path` | Checkpoint output directory |
| `--symbols-per-word` | Complex channel symbols per word; default is `8` |

### Evaluate Original DeepSC

```shell
python evaluation.py --bs=256 --test-snr=6 --channel=AWGN --checkpoint-path=./checkpoint
```

If sentence similarity is enabled in `evaluation.py`, the external BERT model files must be downloaded separately.

## RQ1: Task-Level Semantic Preservation

RQ1 tests whether DeepSC preserves task-level semantics under different SNR values while keeping the original symbol budget and model structure fixed.

RQ1 outputs:

- decoded JSONL files for each method and SNR
- `rq1_summary.csv`
- `rq1_cases.csv`
- BLEU, intent accuracy, slot precision/recall/F1, and task success figures

### Import Amazon MASSIVE

The recommended RQ1 dataset is Amazon MASSIVE `en-US`, imported into the project JSONL schema:

```powershell
python scripts\rq1_import_massive.py --download --locale en-US --output-dir data/rq1_massive --seed 42 --dedupe-text
```

The imported JSONL rows contain:

```json
{"id":"...","split":"train","text":"wake me up at nine am on friday","intent":"alarm_set","slots":{"time":"nine am","date":"friday"}}
```

More dataset details are documented in:

```text
docs/rq1_massive_dataset.md
```

### Run the RQ1 Pipeline

```powershell
python scripts\rq1_run_pipeline.py --stage all --data-root data/rq1_massive --output-dir outputs/rq1_massive --snrs=-6,-3,0,3,6,9,12 --seed 42
```

Quick pipeline check:

```powershell
python scripts\rq1_run_pipeline.py --stage all --quick-test
```

RQ1 trains two model groups:

| Method | Meaning |
|---|---|
| `full` | DeepSC with reconstruction loss and MINE mutual information loss |
| `no_mi` | DeepSC without MINE mutual information loss |

Detailed RQ1 usage is documented in:

```text
experiments/rq1_task_semantics/README.md
```

## RQ2: Task-Level Symbol Efficiency

RQ2 studies how task-level semantic quality changes as the channel-symbol budget changes.

The original DeepSC channel encoder output dimension was fixed at `16`, corresponding to:

```text
symbols_per_word = 8
```

RQ2 trains separate checkpoints for different values such as:

```text
1,2,3,4,6,8,10
```

### Quick Test

```powershell
python scripts\rq2_run_pipeline.py --stage all --quick-test --output-dir outputs/rq2_symbol_efficiency_quick
```

### Full Grid Example

```powershell
python scripts\rq2_run_pipeline.py --stage all --mode grid --symbols-list 1,2,3,4,6,8,10 --snrs -15,-12,-9,-6,-3,0,3,6,9,12 --methods full,no_mi --output-dir outputs/rq2_symbol_efficiency
```

RQ2 produces per-symbol decoded cases, summary metrics, threshold tables, and figures such as task success versus symbols per word and SNR-symbol heatmaps.

Detailed RQ2 usage is documented in:

```text
experiments/rq2_symbol_efficiency/README.md
```

## Experiment Outputs

Typical RQ1 output structure:

```text
outputs/rq1_massive/
  configs/rq1_config.json
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
  logs/
```

Typical RQ2 output structure:

```text
outputs/rq2_symbol_efficiency/
  configs/rq2_config.json
  checkpoints/{method}/spw_{N}/
  decoded/{method}/spw_{N}/snr_{snr}.jsonl
  metrics/
    rq2_summary.csv
    rq2_cases.csv
    rq2_thresholds.csv
  figures/
  logs/
```

## Current Evaluation Notes

The task-level metrics are designed for experimental analysis and paper reporting:

- BLEU-1 through BLEU-4 measure surface text recovery.
- Intent Accuracy measures whether the decoded text preserves the task intent.
- Slot Precision, Slot Recall, and Slot F1 measure whether key task arguments are preserved.
- Task Success Rate requires correct intent and all gold slot values to be preserved.

For MASSIVE, intent evaluation currently uses rule-based logic plus a nearest-original-utterance fallback. It is a practical proxy for task-level semantic preservation. A future extension should replace this with an independently trained NLU classifier.

Slot matching is string-based after normalization. It does not yet handle paraphrases such as `seven am` versus `7 am`.

## Paper-Oriented Intent of This Fork

The added experiment code is intended to support a structured empirical study:

- RQ1 evaluates semantic preservation under channel noise.
- RQ2 evaluates semantic quality under different symbol budgets.
- Amazon MASSIVE provides a more realistic task-oriented benchmark than synthetic templates.
- The original DeepSC commands remain available for baseline reproduction.

Template data remains useful for pipeline sanity checks, but MASSIVE should be treated as the primary RQ1/RQ2 dataset for paper experiments.

## Citation

```bibtex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing},
  title={Deep Learning Enabled Semantic Communication Systems},
  year={2021},
  volume={69},
  pages={2663-2675}
}
```
