import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 0. 仅从 .env 读取：不设业务项的代码兜底默认值，缺失或非法即退出
# ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"

_REQUIRED_KEYS = (
    "MODEL_NAME",
    "DATASET_NAME",
    "DATASET_CONFIG",
    "DATASET_SPLIT",
    "MAX_SAMPLES",
    "DEVICE",
    "BATCH_SIZE",
    "MAX_LENGTH",
    "OUTPUT_DIR",
)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _load_and_validate_env() -> None:
    if not _ENV_FILE.is_file():
        _fail(
            f"未找到 {_ENV_FILE}。\n"
            "请复制模板后显式填写全部变量（程序不为业务配置提供默认值）：\n"
            "  cp .env.example .env"
        )
    load_dotenv(_ENV_FILE, override=False)
    missing = [k for k in _REQUIRED_KEYS if k not in os.environ]
    if missing:
        _fail(
            ".env 中缺少下列键（须全部写出，可参考 .env.example）：\n"
            + "\n".join(f"  - {k}" for k in missing)
        )


_load_and_validate_env()


def _env_nonempty(key: str) -> str:
    raw = os.environ[key]
    s = _strip_quotes(raw).strip()
    if s == "":
        _fail(f"环境变量 {key} 不能为空（请在 {_ENV_FILE} 中填写）。")
    return s


def _parse_max_samples(raw: str) -> Optional[int]:
    s = _strip_quotes(raw).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    try:
        n = int(s)
    except ValueError:
        _fail(f"MAX_SAMPLES 须为正整数，或留空 / none / null 表示全量，当前为：{raw!r}")
    if n < 1:
        _fail(f"MAX_SAMPLES 须 >= 1，或留空 / none / null 表示全量，当前为：{raw!r}")
    return n


def _parse_int(key: str, raw: str) -> int:
    s = _strip_quotes(raw).strip()
    try:
        return int(s)
    except ValueError:
        _fail(f"环境变量 {key} 须为整数，当前为：{raw!r}")


MODEL_NAME = _env_nonempty("MODEL_NAME")
DATASET_NAME = _env_nonempty("DATASET_NAME")

_dc = _env_nonempty("DATASET_CONFIG")
if _dc.strip().lower() in ("none", "null"):
    DATASET_CONFIG: Optional[str] = None
else:
    DATASET_CONFIG = _dc.strip()

MAX_SAMPLES = _parse_max_samples(os.environ["MAX_SAMPLES"])
DATASET_SPLIT = _env_nonempty("DATASET_SPLIT")
DEVICE = _parse_int("DEVICE", os.environ["DEVICE"])
BATCH_SIZE = _parse_int("BATCH_SIZE", os.environ["BATCH_SIZE"])
MAX_LENGTH = _parse_int("MAX_LENGTH", os.environ["MAX_LENGTH"])

_out = _env_nonempty("OUTPUT_DIR")
OUTPUT_DIR = str(_ROOT) if _out.strip() == "." else str(Path(_out).expanduser().resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_LABEL = (
    f"{DATASET_NAME} ({DATASET_CONFIG})"
    if DATASET_CONFIG
    else DATASET_NAME
)

# ──────────────────────────────────────────────
# 1. 加载依赖
# ──────────────────────────────────────────────
print("正在加载依赖...")
print(f"  配置来源: {_ENV_FILE}")
# datasets：从 Hub 加载数据；ClassLabel 用于识别三分类 label 的类别名
from datasets import load_dataset
from datasets.features import ClassLabel
# transformers：AutoTokenizer 分词；AutoModelForSequenceClassification 加载句分类模型；
# pipeline(..., task="sentiment-analysis") 封装批量情感推理
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
# sklearn.metrics：与真实 label 对比，计算准确率/精确率/召回率/F1、混淆矩阵与分类报告
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import numpy as np


def _prepare_binary_sentiment(ds):
    """三分类（含 neutral）时剔除中性并映射为二分类 0/1；优先用 ClassLabel 元数据（label 为整数时）。"""
    lab_feat = ds.features.get("label")
    if isinstance(lab_feat, ClassLabel) and lab_feat.names:
        names = [str(n).lower() for n in lab_feat.names]
        if "neutral" in names and "positive" in names and "negative" in names:
            neu_i = names.index("neutral")
            pos_i = names.index("positive")
            before = len(ds)
            ds = ds.filter(lambda ex: ex["label"] != neu_i)
            ds = ds.map(lambda ex: {"label": 1 if ex["label"] == pos_i else 0})
            print(
                f"  （三分类 ClassLabel）已剔除 neutral：{before} → {len(ds)} 条，"
                "label→0=negative / 1=positive"
            )
            return ds

    if "label_text" not in ds.column_names:
        return ds
    try:
        uniq = {str(x).lower() for x in ds.unique("label_text")}
    except Exception:
        uniq = {str(ds[i]["label_text"]).lower() for i in range(min(500, len(ds)))}
    if "neutral" not in uniq:
        return ds
    before = len(ds)
    ds = ds.filter(lambda ex: str(ex["label_text"]).lower() != "neutral")
    ds = ds.map(
        lambda ex: {
            "label": 1 if str(ex["label_text"]).lower() == "positive" else 0,
        }
    )
    print(
        f"  （三分类 label_text）已剔除 neutral：{before} → {len(ds)} 条，"
        "label→0=negative / 1=positive"
    )
    return ds


print(f"模型: {MODEL_NAME}")
print(f"数据集: {DATASET_LABEL} | split={DATASET_SPLIT}")
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("─" * 50)

# ──────────────────────────────────────────────
# 2. 加载数据集
# ──────────────────────────────────────────────
print("\n[Step 1/4] 加载数据集...")
raw_dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split=DATASET_SPLIT,
)
raw_dataset = _prepare_binary_sentiment(raw_dataset)

if MAX_SAMPLES:
    n = min(MAX_SAMPLES, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(n))

print(f"  样本数: {len(raw_dataset)}")
print(f"  示例: {raw_dataset[0]['text'][:50]}...")

label_counts: dict[str, int] = {}
if "label_text" in raw_dataset.column_names:
    for lb in raw_dataset["label_text"]:
        k = str(lb)
        label_counts[k] = label_counts.get(k, 0) + 1
else:
    for lb in raw_dataset["label"]:
        k = str(lb)
        label_counts[k] = label_counts.get(k, 0) + 1
print(f"  标签统计: {label_counts}")

# ──────────────────────────────────────────────
# 3. 加载模型和分词器
# ──────────────────────────────────────────────
print("\n[Step 2/4] 加载模型和分词器...")
print("  （首次运行会下载模型，约 430MB）")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    truncation=True,
    max_length=MAX_LENGTH,
)
print("  ✓ 模型加载完成")

# ──────────────────────────────────────────────
# 4. 批量推理
# ──────────────────────────────────────────────
print("\n[Step 3/4] 批量推理中...")
predictions = []
labels = []
texts = []

batch_size = BATCH_SIZE
total = len(raw_dataset)
start_time = time.time()

for i in range(0, total, batch_size):
    batch = raw_dataset.select(range(i, min(i + batch_size, total)))
    texts.extend(batch["text"])

    results = sentiment_pipeline(batch["text"])

    for result in results:
        pred_label = result["label"].lower()
        predictions.append(1 if pred_label == "positive" else 0)

    labels.extend(batch["label"])

    done = min(i + batch_size, total)
    elapsed = time.time() - start_time
    speed = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / speed if speed > 0 else 0
    print(f"\r  进度: {done}/{total} ({100*done/total:.0f}%) "
          f"| 速度: {speed:.1f}样本/秒  |  ETA: {eta:.0f}秒",
          end="", flush=True)

elapsed_total = time.time() - start_time
print(f"\n  ✓ 推理完成，耗时: {elapsed_total:.1f}秒")

# ──────────────────────────────────────────────
# 5. 计算评测指标
# ──────────────────────────────────────────────
print("\n[Step 4/4] 计算评测指标...")

y_true = np.array(labels)
y_pred = np.array(predictions)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="binary", pos_label=1, zero_division=0
)
recall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
)

label_names = ["negative", "positive"]

# ──────────────────────────────────────────────
# 6. 打印结果
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("               评测报告")
print("=" * 55)
print(f"  模型     : {MODEL_NAME}")
print(f"  数据集   : {DATASET_LABEL} | split={DATASET_SPLIT}")
print(f"  样本数   : {len(raw_dataset)}")
print(f"  耗时     : {elapsed_total:.1f}秒")
print("─" * 55)
print(f"  Accuracy  (准确率) : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision (精确率): {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    (召回率): {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1 Score  (F1值)  : {f1:.4f}  ({f1*100:.2f}%)")
print("─" * 55)
print("  混淆矩阵（行=真实标签，列=预测标签）:")
print(f"               预测negative  预测positive")
print(f"  真实negative    {cm[0][0]:>5}        {cm[0][1]:>5}")
print(f"  真实positive    {cm[1][0]:>5}        {cm[1][1]:>5}")
print("─" * 55)
print("  详细分类报告:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=label_names,
        zero_division=0,
    )
)
print("=" * 55)

# ──────────────────────────────────────────────
# 7. 保存结果
# ──────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_slug = MODEL_NAME.replace("/", "_")
report = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "dataset": DATASET_LABEL,
    "dataset_split": DATASET_SPLIT,
    "num_samples": len(raw_dataset),
    "elapsed_seconds": round(elapsed_total, 2),
    "metrics": {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    },
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized": cm_normalized.tolist(),
}
report_path = os.path.join(OUTPUT_DIR, f"eval_report_{model_slug}_{timestamp}.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n报告已保存: {report_path}")

# ──────────────────────────────────────────────
# 8. 错误样本分析
# ──────────────────────────────────────────────
print("\n【错误样本分析】—— 预测错误的样本（前 5 条）:")
error_count = 0
for i in range(len(texts)):
    if y_true[i] != y_pred[i]:
        error_count += 1
        if error_count <= 5:
            true_l = label_names[y_true[i]]
            pred_l = label_names[y_pred[i]]
            print(f"\n  [{error_count}] 文本: {texts[i][:80]}...")
            print(f"       真实: {true_l}  |  预测: {pred_l}")

print(f"\n总错误数: {error_count}/{len(texts)} ({100*error_count/len(texts):.1f}%)")
print("\n✓ 评测完成！")
