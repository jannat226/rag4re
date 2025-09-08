import json 
from itertools import islice

train_path  = '/home/lnuj3/thesis/gutbrainie2025/Annotations/Train/platinum_quality/json_format/train_platinum.json'
with open(train_path,'r') as file:
    full_train = json.load(file)

# small_train = full_train[:100]
# keep first 100 key-value pairs
small_train = dict(islice(full_train.items(), 100))

with open('small_train.json','w') as file:
    json.dump(small_train, file ,indent = 2)
