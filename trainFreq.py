import json
from collections import Counter

# Load the train data
with open("processed_train.json", "r") as f:
    train_data = json.load(f)

# Normalize and extract all relation labels
def normalize_relation(label):
    return str(label).strip().upper() if label is not None else "NONE"

relations = [normalize_relation(record["relation"]) for record in train_data]

# Count frequencies of each relation
frequency = Counter(relations)

# Convert to a dict for pretty saving
freq_dict = dict(frequency)

# Save frequency counts to a file
with open("train_relation_frequency.json", "w") as out_f:
    json.dump(freq_dict, out_f, indent=4)

print("Frequency saved to relation_frequency.json")
