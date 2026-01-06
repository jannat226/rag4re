import json
import os
from collections import Counter
from io import StringIO
from sklearn.metrics import precision_recall_fscore_support


def normalize_label(label):
    return str(label).strip().lower()


# Paths
predictions_dir = "predictions"
predictions_file = "rag4re_predictions_5_shot_non_reasoning_qwen.json"
predictions_path = os.path.join(predictions_dir, predictions_file)

# Load predictions
with open(predictions_path, "r") as f:
    outputs = json.load(f)

# Collect predictions and ground truths
all_predictions = [normalize_label(o["prediction"]) for o in outputs]
all_groundtruths = [normalize_label(o["ground_prediction"]) for o in outputs]

# Frequency counts
pred_counter = Counter(all_predictions)
gt_counter = Counter(all_groundtruths)

# -------- capture text in buffer instead of printing directly --------
buf = StringIO()
w = buf.write

w("Ground truth class frequencies:\n")
for label, count in gt_counter.items():
    w(f"{label}: {count}\n")
w(f"Total unique ground truth labels: {len(gt_counter)}\n")

w("\nPrediction class frequencies:\n")
for label, count in pred_counter.items():
    w(f"{label}: {count}\n")
w(f"Total unique predicted labels: {len(pred_counter)}\n")

# All unique labels (union of true and predicted)
all_labels = sorted(set(all_groundtruths) | set(all_predictions))
w(f"\nTotal unique labels (union): {len(all_labels)}\n")

# Per-label stats over the full union
precision, recall, f1, support = precision_recall_fscore_support(
    all_groundtruths,
    all_predictions,
    labels=all_labels,
    zero_division=0,
)

# Per-class metrics
w("\nPer-label scores (over union of labels):\n")
w(f"{'Label':30s} {'F1':>6s} {'Precision':>9s} {'Recall':>7s} {'Support':>8s}\n")
for label, f1_val, p, r, sup in zip(all_labels, f1, precision, recall, support):
    w(f"{label:<25s} {f1_val:6.3f} {p:9.3f} {r:7.3f} {sup:8d}\n")

w("\nLabels considered for per-label computation (union):\n")
w(f"{all_labels}\n")
w(f"Number of classes in union: {len(all_labels)}\n")

# ----------------- Macro metrics over ground-truth labels only -----------------
gt_labels = sorted(gt_counter.keys())
w("\nGround truth label subset used for macro metrics:\n")
w(f"{gt_labels}\n")
w(f"Number of ground truth classes: {len(gt_labels)}\n")

# Map label -> per-label precision/recall/F1 (computed over union)
label_to_precision = dict(zip(all_labels, precision))
label_to_recall = dict(zip(all_labels, recall))
label_to_f1 = dict(zip(all_labels, f1))

# Collect metrics only for labels present in ground truth
prec_gt_labels = [label_to_precision[label] for label in gt_labels]
recall_gt_labels = [label_to_recall[label] for label in gt_labels]
f1_gt_labels = [label_to_f1[label] for label in gt_labels]

macro_precision_gt = sum(prec_gt_labels) / len(prec_gt_labels) if prec_gt_labels else 0.0
macro_recall_gt = sum(recall_gt_labels) / len(recall_gt_labels) if recall_gt_labels else 0.0
macro_f1_gt = sum(f1_gt_labels) / len(f1_gt_labels) if f1_gt_labels else 0.0

w("\nMacro metrics over ground truth labels only:\n")
w(f"  Macro Precision (ground_truth_only): {macro_precision_gt:.4f}\n")
w(f"  Macro Recall    (ground_truth_only): {macro_recall_gt:.4f}\n")
w(f"  Macro F1        (ground_truth_only): {macro_f1_gt:.4f}\n")

# ----------------- Micro metrics (over all labels) -----------------
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
    all_groundtruths,
    all_predictions,
    labels=all_labels,
    average="micro",
    zero_division=0,
)

w("\nMicro metrics (over all labels):\n")
w(f"  Micro Precision: {micro_p:.4f}\n")
w(f"  Micro Recall:    {micro_r:.4f}\n")
w(f"  Micro F1 score:  {micro_f1:.4f}\n")

# -------- save buffer contents to a file --------
report_text = buf.getvalue()

output_dir = "evaluation_metrics"
os.makedirs(output_dir, exist_ok=True)

output_file = "rag4re_5_shot_non_reasoning_qwen_metrics.txt"
output_path = os.path.join(output_dir, output_file)

with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write(report_text)

print(f"\nSaved evaluation report to: {output_path}")
