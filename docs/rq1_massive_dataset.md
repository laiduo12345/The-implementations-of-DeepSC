# RQ1 数据集说明：Amazon MASSIVE

## 1. 数据集概述

本项目在 RQ1（Task-level Semantic Preservation under Different SNRs）中引入 Amazon MASSIVE 数据集，作为任务型语义通信实验的主要真实数据来源。MASSIVE 是面向虚拟助手场景的多语言自然语言理解数据集，包含用户指令、意图标签和槽位标注。当前实验使用其中的英文 `en-US` 子集，并将其转换为本项目统一的 JSONL 格式。

与前期模板数据相比，MASSIVE 的优势在于其表达形式更接近真实用户指令，覆盖更多任务域和意图类型，并提供官方 train/dev/test 划分。它能够更合理地评估 DeepSC 在不同信噪比条件下对任务级语义信息的保持能力。

## 2. 数据来源与导入方式

| 项目 | 内容 |
|---|---|
| 数据集名称 | Amazon MASSIVE |
| 使用语言 | English, `en-US` |
| 数据类型 | 单轮虚拟助手自然语言理解数据 |
| 样本字段 | utterance, intent, slot annotation, scenario, partition |
| 本项目导入脚本 | `scripts/rq1_import_massive.py` |
| 本项目输出目录 | `data/rq1_massive/` |
| 本项目统一格式 | `id`, `split`, `text`, `intent`, `slots`, `scenario`, `source` |

导入命令示例：

```powershell
.venv\Scripts\python.exe scripts\rq1_import_massive.py `
  --download `
  --locale en-US `
  --output-dir data/rq1_massive `
  --seed 42 `
  --dedupe-text
```

导入后可直接接入 RQ1 pipeline：

```powershell
.venv\Scripts\python.exe scripts\rq1_run_pipeline.py `
  --stage all `
  --data-root data/rq1_massive `
  --output-dir outputs/rq1_massive `
  --snrs=-6,-3,0,3,6,9,12 `
  --seed 42
```

## 3. 当前导入数据统计

当前 `data/rq1_massive/` 已完成导入，统计结果如下。

| 指标 | 数值 |
|---|---:|
| 总样本数 | 16,428 |
| Train 样本数 | 11,454 |
| Valid 样本数 | 2,025 |
| Test 样本数 | 2,949 |
| 唯一文本数 | 16,428 |
| 重复文本数 | 0 |
| Test 与 Train 精确文本重合数 | 0 |
| Intent 类型数 | 60 |
| Scenario 类型数 | 18 |
| Slot 类型数 | 55 |
| 句长中位数 | 约 6 words |
| 平均句长 | 约 6.8-6.9 words |
| 90% 分位句长 | 约 11 words |
| 无 slot 样本比例 | 约 32%-34% |

### 3.1 数据划分

| Split | 样本数 | 句长中位数 | 平均句长 | 90% 分位句长 | 平均 Slot 数 | 无 Slot 比例 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 11,454 | 6 | 6.93 | 11 | 0.99 | 32.6% |
| Valid | 2,025 | 6 | 6.88 | 11 | 0.99 | 31.8% |
| Test | 2,949 | 6 | 6.79 | 11 | 0.95 | 33.5% |

### 3.2 Top Intent 分布

| Intent | 样本数 |
|---|---:|
| `calendar_set` | 1,146 |
| `play_music` | 934 |
| `weather_query` | 848 |
| `general_quirky` | 825 |
| `calendar_query` | 793 |
| `qa_factoid` | 768 |
| `news_query` | 707 |
| `email_query` | 604 |
| `email_sendemail` | 530 |
| `datetime_query` | 493 |

### 3.3 Scenario 分布

| Scenario | 样本数 |
|---|---:|
| `calendar` | 2,361 |
| `play` | 2,016 |
| `qa` | 1,673 |
| `email` | 1,374 |
| `iot` | 1,104 |
| `general` | 957 |
| `weather` | 848 |
| `transport` | 800 |
| `lists` | 789 |
| `news` | 707 |
| `recommendation` | 593 |
| `datetime` | 569 |
| `social` | 558 |
| `alarm` | 549 |
| `music` | 468 |
| `audio` | 385 |
| `takeaway` | 355 |
| `cooking` | 322 |

### 3.4 高频 Slot 类型

| Slot 类型 | 出现次数 |
|---|---:|
| `date` | 2,561 |
| `place_name` | 1,557 |
| `event_name` | 1,417 |
| `person` | 1,211 |
| `time` | 1,125 |
| `media_type` | 695 |
| `business_name` | 522 |
| `weather_descriptor` | 459 |
| `transport_type` | 432 |
| `food_type` | 413 |

## 4. 数据样例

| Scenario | Intent | Text | Slots |
|---|---|---|---|
| alarm | `alarm_set` | `wake me up at nine am on friday` | `date=friday`, `time=nine am` |
| cooking | `cooking_recipe` | `how long should i boil the eggs` | `cooking_type=boil`, `food_type=eggs` |
| weather | `weather_query` | `is it going to rain` | `weather_descriptor=rain` |
| play | `play_music` | `i want to listen arijit singh song once again` | `artist_name=arijit singh` |
| recommendation | `recommendation_locations` | `find me a nice restaurant for dinner` | `business_type=restaurant`, `meal_type=dinner` |
| transport | `transport_query` | `you need to give me different directions` | none |

## 5. 数据流与实验位置

```mermaid
flowchart LR
    A[Amazon MASSIVE en-US] --> B[rq1_import_massive.py]
    B --> C[data/rq1_massive/*.jsonl]
    C --> D[rq1_convert_data.py]
    D --> E[DeepSC pickle + vocab]
    E --> F[rq1_train_deepsc.py]
    F --> G[full / no_mi checkpoints]
    G --> H[rq1_decode_snr.py]
    H --> I[decoded JSONL by SNR]
    I --> J[rq1_evaluate_task_metrics.py]
    J --> K[BLEU / Intent / Slot / Task Success]
```

## 6. 与 RQ1 的适配性分析

MASSIVE 适合作为 RQ1 主实验数据集，原因如下。

1. **任务级语义标签完整**  
   每条样本包含自然语言指令、意图标签和槽位标注，能够直接支持 Intent Accuracy、Slot Recall、Slot F1 和 Task Success Rate 等任务级指标。

2. **任务域覆盖广泛**  
   数据覆盖 18 个 scenario 和 60 个 intent，包括 alarm、calendar、weather、music、transport、email、iot、qa 等常见虚拟助手任务。相比人工模板数据，其任务语义空间更大。

3. **减少模板重复导致的虚高指标**  
   当前导入版本中，所有文本均唯一，且 test 与 train 没有精确文本重合。该性质能够缓解前期模板数据中 train/test 大量重复导致的 SNR=0 条件下指标异常偏高问题。

4. **槽位保留评估更有意义**  
   数据包含 55 类 slot，且 test split 中存在部分 train 未见过的 slot value。因此，模型不仅需要恢复句式，还需要在噪声信道下尽量保持关键实体、时间、地点、对象等槽位信息。

5. **句长适合当前 DeepSC 结构**  
   MASSIVE 英文子集大多数句子长度在 5-11 words 范围内，符合当前 DeepSC 默认最大长度和训练设置，工程接入成本较低。

## 7. 局限性与实验注意事项

尽管 MASSIVE 明显优于模板数据，但在论文实验设计中仍需客观说明以下限制。

| 限制 | 说明 | 可能影响 |
|---|---|---|
| 单轮任务数据 | MASSIVE 主要是单轮虚拟助手指令 | 不覆盖多轮上下文语义保持 |
| 句子整体较短 | 中位数约 6 words | 对长句语义通信能力考察不足 |
| 约三分之一样本无 slot | audio、general、transport 等场景中存在无槽位样本 | Slot 指标对部分样本不敏感 |
| Intent 评估当前为启发式 | 本项目对非模板 intent 使用 nearest-original-utterance fallback | Intent Accuracy 是 proxy，不等同于独立 NLU 分类器结果 |
| Slot 匹配为字符串级别 | 当前只做规范化字符串匹配 | 无法识别同义表达或数字形式变体 |

## 8. 建议的实验设置

### 8.1 主实验：全量 MASSIVE

主实验建议使用全部导入样本，以评估 DeepSC 在真实任务型指令分布上的整体语义保持能力。

```powershell
.venv\Scripts\python.exe scripts\rq1_run_pipeline.py `
  --stage all `
  --data-root data/rq1_massive `
  --output-dir outputs/rq1_massive `
  --snrs=-6,-3,0,3,6,9,12 `
  --seed 42
```

### 8.2 Slot-focused 实验：仅保留有 slot 的样本

为了更集中评估槽位实体的保持能力，可构造 `min_slots >= 1` 的子集。

```powershell
.venv\Scripts\python.exe scripts\rq1_import_massive.py `
  --source data\raw\massive\amazon-massive-dataset-1.1.tar.gz `
  --output-dir data/rq1_massive_slots `
  --min-slots 1 `
  --seed 42
```

对应 pipeline：

```powershell
.venv\Scripts\python.exe scripts\rq1_run_pipeline.py `
  --stage all `
  --data-root data/rq1_massive_slots `
  --output-dir outputs/rq1_massive_slots `
  --snrs=-6,-3,0,3,6,9,12 `
  --seed 42
```

## 9. 论文写作建议

在论文中，建议将当前模板数据定位为 pipeline sanity check，而将 MASSIVE 定位为 RQ1 的主要实验数据集。相关表述应避免声称 MASSIVE 能覆盖全部自然语言语义通信场景，更准确的说法是：

> 本文采用 Amazon MASSIVE 英文子集构造任务型语义通信评估集。该数据集覆盖多个虚拟助手任务域，并提供 intent 与 slot 标注，适合评估通信系统在不同信道条件下对任务级语义信息的保持能力。

同时，建议在实验局限性中说明：

> 当前任务成功率基于规则与字符串匹配实现，反映的是可解释的任务语义保留情况；后续工作可引入独立训练的 NLU 模型作为语义判别器，以进一步提升评估指标的鲁棒性。
