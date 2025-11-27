import json
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import wandb

# Utility for normalization
def normalize_label(label):
    # Lowercase and strip spaces, handle None safely
    return str(label).strip().lower() if label is not None else "none"

# Load predictions and ground truths
with open("rag4re_predictions_10shot_qwen.json", "r") as f:
    outputs = json.load(f)

# Normalize all labels
all_predictions = [normalize_label(o["prediction"]) for o in outputs]
all_groundtruths = [normalize_label(o["ground_prediction"]) for o in outputs]

# Frequency counts
pred_counter = Counter(all_predictions)
gt_counter = Counter(all_groundtruths)

# Union of all unique labels
all_labels = sorted(set(all_groundtruths) | set(all_predictions))

# Print diagnostic info
print("Unique groundtruth labels:", set(all_groundtruths))
print("Unique prediction labels:", set(all_predictions))
print("Labels in all_labels:", all_labels)
missing = set(all_groundtruths + all_predictions) - set(all_labels)
print("Missing from all_labels (should be empty):", missing)

# Table: Ground truth vs Prediction count per label
print(f"\n{'Label':<25} {'Ground Truth':>12} {'Prediction':>12}")
for label in all_labels:
    print(f"{label:<25} {gt_counter.get(label, 0):12d} {pred_counter.get(label, 0):12d}")

# Per-label metrics
precision, recall, f1, support = precision_recall_fscore_support(
    all_groundtruths, all_predictions, labels=all_labels, zero_division=0
)

print("\nPer-label Scores:")
print(f"{'Label':<25} {'F1':>6} {'Precision':>9} {'Recall':>7} {'Support':>8}")
for label, f1_val, p, r, sup in zip(all_labels, f1, precision, recall, support):
    print(f"{label:<25} {f1_val:6.3f} {p:9.3f} {r:7.3f} {sup:8d}")

# Compute the confusion matrix (with union of all labels as axis)
cm = confusion_matrix(all_groundtruths, all_predictions, labels=all_labels)

# Log to wandb with every label included, including 'none'
wandb.init(project="your_project_name")  # <-- Replace with your actual project
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_groundtruths,
        preds=all_predictions,
        class_names=all_labels
    )
})
wandb.finish()
