import json
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

def normalize_label(label):
    return str(label).strip().lower()

with open("rag4re_predictions_35shot_qwen.json", "r") as f:
    outputs = json.load(f)

# Collect predictions and ground truths
all_predictions = [normalize_label(o["prediction"]) for o in outputs]
all_groundtruths = [normalize_label(o["ground_prediction"]) for o in outputs]

# Frequency counts
pred_counter = Counter(all_predictions)
gt_counter = Counter(all_groundtruths)

print("Ground truth class frequencies:")
for label, count in gt_counter.items():
    print(f"{label}: {count}")
print(f"Total unique ground truth labels: {len(gt_counter)}")

print("\nPrediction class frequencies:")
for label, count in pred_counter.items():
    print(f"{label}: {count}")
print(f"Total unique predicted labels: {len(pred_counter)}")

# Find ALL unique labels for full per-class F1/precision/recall (union of true and predicted)
all_labels = sorted(set(all_groundtruths) | set(all_predictions))

print(f"\nTotal unique labels (union): {len(all_labels)}")

# Compute per-label stats
precision, recall, f1, support = precision_recall_fscore_support(
    all_groundtruths, all_predictions, labels=all_labels, zero_division=0
)

# Print per-class metrics
print("\nPer-label Scores:")
print(f"{'Label':<25s} {'F1':>6s} {'Precision':>9s} {'Recall':>7s} {'Support':>8s}")
for label, f1_val, p, r, sup in zip(all_labels, f1, precision, recall, support):
    print(f"{label:<25s} {f1_val:6.3f} {p:9.3f} {r:7.3f} {sup:8d}")


print("\nLabels considered for macro F1 computation:")
print(all_labels)
print(f"Number of classes: {len(all_labels)}")

macro_f1 = f1.mean()
print(f"\nMacro F1 score (mean of per-class F1s): {macro_f1:.4f}")



# Get just the ground truth label set
gt_labels = sorted(gt_counter.keys())

# Get index mapping from all_labels to per-class scores
label_to_f1 = dict(zip(all_labels, f1))

# Gather the F1 scores only for labels present in ground truth
f1_gt_labels = [label_to_f1[label] for label in gt_labels]

# Compute the average F1 over just ground truth labels
macro_f1_gt = sum(f1_gt_labels) / len(f1_gt_labels) if f1_gt_labels else 0.0

print(f"\nMean F1 over ground truth labels only: {macro_f1_gt:.4f}")
print("Ground truth label subset used for mean F1:")
print(gt_labels)


# Compute micro F1 score
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
    all_groundtruths, all_predictions, labels=all_labels, average='micro', zero_division=0
)
print(f"\nMicro Precision: {micro_p:.4f}")
print(f"Micro Recall: {micro_r:.4f}")
print(f"Micro F1 score: {micro_f1:.4f}")