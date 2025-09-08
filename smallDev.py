import json 
dev_path  = '/home/lnuj3/thesis/gutbrainie2025/Annotations/Dev/json_format/dev.json'

with open(dev_path, 'r', encoding = 'utf-8') as file :
    full_dev_data = json.load(file)
small_dev_data =  dict(list(full_dev_data.items())[:5])

with open('dev5.json', 'w') as fw:
    json.dump(small_dev_data, fw, indent=4)
