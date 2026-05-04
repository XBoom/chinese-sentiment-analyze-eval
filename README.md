# 中文情感分析评测（Chinese Sentiment Analyze Eval）

在 Hugging Face 数据集上，用预训练的中文情感分类模型做**批量推理**，并计算 **Accuracy / Precision / Recall / F1**、混淆矩阵与分类报告；终端打印摘要，同时在项目目录生成 **`eval_report_*.json`** 便于对比实验。

---

## 项目做什么

| 项目 | 说明 |
|------|------|
| **输入** | 数据集中的 `text` 字段（中文商品评论等） |
| **模型** | 默认 `uer/roberta-base-finetuned-jd-binary-chinese`（序列分类 + `sentiment-analysis` pipeline） |
| **标签** | 二分类：`label` 为 **0=negative、1=positive**（与当前评测脚本一致） |
| **输出** | 终端报告 + JSON；可选打印若干条预测错误样例 |

脚本不训练模型，只做**加载 → 推理 → 指标统计**，适合快速对比不同模型或不同数据划分。

**数据处理顺序**：`datasets.load_dataset` →（若为含 `neutral` 的三分类，则按 `ClassLabel` 或 `label_text` **剔除中性**并把 `label` 规范为 0/1）→ 再按 `MAX_SAMPLES` 截断 → 推理与打指标。

**配置加载**：环境变量由项目根目录的 **`.env`** 提供，通过 **`python-dotenv`** 的 `load_dotenv()` 加载（见 `requirements.txt`）。**`.env.example`** 仅作模板，脚本不会读取它。

---

## 仓库结构

```
.
├── evaluate_chinese_sentiment.py   # 主入口：评测脚本
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板（可复制为 .env）
├── README.md
└── .venv/                          # 本地虚拟环境（创建后生成，勿提交）
```

---

## 环境要求

- **Python**：建议 **3.10～3.12**。若系统默认是 **3.13/3.14**，PyPI 上可能没有对应 **torch** wheel，或与本项目锁定的 `transformers` 版本不兼容；请用已安装好的 **3.11** 创建虚拟环境，例如：  
  `$(command -v python3.11) -m venv .venv`（路径因机器而异，Homebrew / pyenv / Conda 均可）。
- **磁盘**：首次运行需下载模型（约 **430MB**）与数据集，请预留空间。
- **网络**：需能访问 **Hugging Face**（下载模型与 `datasets`）；受限网络可配置镜像或 `HF_ENDPOINT` 等（见 Hugging Face 文档）。
- **硬件**：默认 **CPU**（`DEVICE=-1`）；有 NVIDIA GPU 且安装 CUDA 版 PyTorch 时，可在 `.env` 中将 `DEVICE` 设为 `0` 等以加速。

---

## 虚拟环境（推荐）

在项目根目录执行（将 `/path/to/...` 换成你的本机路径）：

```bash
cd /path/to/chinese-sentiment-analyze-eval

# 若 `python3` 版本过新，请显式指定 3.10～3.12，例如：
# python3.11 -m venv .venv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt
```

确认依赖与解释器一致：

```bash
python -c "import torch, transformers, datasets; print(torch.__version__)"
```

运行评测：

```bash
python evaluate_chinese_sentiment.py
```

退出虚拟环境：`deactivate`。

### `requirements.txt` 版本说明（摘要）

| 约束 | 原因 |
|------|------|
| `torch<2.3` | 部分 macOS x86_64 环境 PyPI 上常见最高为 2.2.x |
| `transformers<4.45` | 较新版本可能要求 `torch>=2.6` 才允许 `torch.load` 加载部分权重；在 torch 2.2 下锁定 4.44.x |
| `datasets<3` | `datasets` 3.x/4.x 对部分 Hub 上仍带 **dataset 脚本** 的数据集不兼容 |
| `numpy<2` | 与 torch 2.2 组合时，numpy 2.x 易出现二进制不兼容告警 |
| `tokenizers<0.20` | 与 `transformers 4.44` 匹配 |

若你已升级到 **torch ≥ 2.6**，可自行放宽上表中的上限并在本地验证。

---

## 使用 Conda（可选）

若更习惯 Conda：

```bash
conda create -n chinese-sentiment-eval python=3.10 -y
conda activate chinese-sentiment-eval
pip install -r requirements.txt
python evaluate_chinese_sentiment.py
```

---

## 配置说明（`.env`）

脚本**只**读取项目根目录下的 **`.env`**。下列键**必须全部出现**（可参考 `.env.example` 逐项复制）；**程序不为这些业务项提供代码兜底默认值**——缺少任一键、或某键为空（另有说明的除外）会直接退出，避免「没配环境却默默跑默认配置」。

`load_dotenv(..., override=False)`：已在 shell 里 `export` 的变量**不会被** `.env` 覆盖。

**显式约定**：`MAX_SAMPLES` 键必须存在；值为**空**、`none`、`null` 表示该 split **全量**；否则须为 **≥1** 的整数。`DATASET_CONFIG` 写 **`none`** 表示 `load_dataset` 不传 config。`OUTPUT_DIR` 写 **`.`** 表示报告写在脚本所在目录。

首次使用建议：

```bash
cp .env.example .env
# 再编辑 .env
```

| 变量 | 含义 |
|------|------|
| `MODEL_NAME` | Hugging Face 模型 ID |
| `DATASET_NAME` | 数据集 ID |
| `DATASET_CONFIG` | 多配置数据集的子集名；不需要时可设为 `none` |
| `DATASET_SPLIT` | 如 `test`、`validation`、`train` |
| `MAX_SAMPLES` | 留空或 `none` 表示该 split 全量；正整数表示只取前 N 条（试跑） |
| `DEVICE` | `-1`：CPU；`0` 起：GPU 编号 |
| `BATCH_SIZE` | 批大小（显存不足可调小） |
| `MAX_LENGTH` | tokenizer 截断长度 |
| `OUTPUT_DIR` | 报告 JSON 目录；填 **`.`** 为脚本所在目录（须显式写出，不能省略该键） |

更换数据集时，需仍能通过 `datasets.load_dataset` 加载，且包含 **`text`** 与 **`label`**。若为**原生二分类**（0/1），可直接评测；若为**含 neutral 的三分类**且 `label` 为带 `ClassLabel.names` 的整数，脚本会自动剔除 neutral 并映射为 0/1。有 **`label_text`** 时用于统计展示；没有也可运行。

更换模型时，模型应适合 `pipeline(task="sentiment-analysis")`，且返回的 `label` 经小写后为 **`positive`** 或 **`negative`**；否则需在脚本中自行改解析逻辑。

---

## 运行结果

- **终端**：评测指标、混淆矩阵、`classification_report`、最多 5 条错误样例。
- **文件**：`eval_report_<模型名替换斜杠>_<时间戳>.json`（默认在脚本目录，或 `OUTPUT_DIR`）。

---

## 默认数据集：`tyqiangz/multilingual-sentiments`

| 属性 | 说明 |
|------|------|
| **语言** | 多语言子集；中文子集配置名为 **`chinese`**（全小写，与 Hub 上 `BuilderConfig` 一致） |
| **任务** | Hub 描述为 **positive / neutral / negative** 三分类；本脚本会 **去掉 neutral** 后按 `ClassLabel` 映射为二分类再与模型对比 |
| **来源** | Amazon 多语言评论等汇总 |
| **规模** | 各子集行数以 **Hugging Face 数据集页面 / 实际 `load_dataset` 为准**（版本更新后可能与旧文档不一致） |

字段示例（以实际加载为准；`label` 在映射前可能为 0/1/2 等）：

```json
{
  "text": "这个产品非常好用，值得购买。",
  "label": 1,
  "label_text": "positive"
}
```

### 其他可选数据集（需自行对齐字段与标签）

| 数据集 | 说明 |
|--------|------|
| `SimFoot/ChineseProductReviewSentiment` | 中文商品评论情感 |
| `ShengdingHu/CLUECorpus2020` | 大规模中文语料（需自行构造标签任务） |
| `mteb/amazon_reviews_multi` | 多语言 Amazon 评论 |
| `uer/chnsenticorp` | 中文酒店评论情感 |

---

## 默认模型：`uer/roberta-base-finetuned-jd-binary-chinese`

| 属性 | 说明 |
|------|------|
| **基座** | 中文 RoBERTa-wwm-ext |
| **微调数据** | 京东评论约 20 万条 |
| **任务** | 中文二分类情感 |
| **大小** | 约 430 MB |

### 其他可选模型（替换前请确认 pipeline 标签格式）

| 模型 | 说明 |
|------|------|
| `hfl/chinese-roberta-wwm-ext` | 基座，需微调后才能直接评测本任务 |
| `hfl/chinese-bert-52` | 更大中文 BERT |
| `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` | Erlangshen 情感 |
| `shibing624/text2vec-base-chinese-sentence` | 句向量，非本脚本默认定式 |

在 `.env` 中修改 `MODEL_NAME` 即可切换（需满足上文 pipeline 与标签约定）。

---

## 常见问题

1. **`ModuleNotFoundError`**  
   确认已 `source .venv/bin/activate`（或等价激活），并在**同一解释器**下执行 `pip install -r requirements.txt`。不确定时用：`python -m pip install -r requirements.txt`。

2. **下载慢或超时**  
   检查网络与代理；可设置 Hugging Face 相关环境变量或使用镜像（以官方文档为准）。

3. **CUDA / GPU**  
   需安装与显卡匹配的 PyTorch CUDA 版本，并将 `.env` 中 `DEVICE` 设为 `0` 等。

4. **显存不足**  
   减小 `BATCH_SIZE` 或 `MAX_LENGTH`，或先用 `MAX_SAMPLES` 小规模试跑。

5. **`ValueError: BuilderConfig 'Chinese' not found`**  
   子集名需与 Hub 完全一致，中文子集应为 **`chinese`**。

6. **`RuntimeError: Dataset scripts are no longer supported`**  
   多为 `datasets` 版本过新，请保持 `pip install -r requirements.txt` 中的 **`datasets<3`**。

7. **试跑时 `MAX_SAMPLES` 很小、真实标签只有一类**  
   sklearn 可能提示单类标签；脚本已对混淆矩阵与 `classification_report` 固定 `labels=[0,1]`，并设置 `zero_division=0`，可正常结束。全量数据上评测更有意义。

---

## 许可证

见仓库根目录 `LICENSE`。
