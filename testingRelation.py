import json

# 1. Define the valid_relations mapping.
valid_relations = {
    ("anatomical location", "human"): "located in",
    ("anatomical location", "animal"): "located in",
    ("bacteria", "bacteria"): "interact",
    ("bacteria", "chemical"): "interact",
    ("bacteria", "drug"): "interact",
    ("bacteria", "DDF"): "influence",
    ("bacteria", "gene"): "change expression",
    ("bacteria", "human"): "located in",
    ("bacteria", "animal"): "located in",
    ("bacteria", "microbiome"): "part of",
    ("chemical", "anatomical location"): "located in",
    ("chemical", "human"): "located in",
    ("chemical", "animal"): "located in",
    ("chemical", "chemical"): "interact",
    ("chemical", "microbiome"): "impact",
    ("chemical", "bacteria"): "impact",
    ("dietary supplement", "bacteria"): "impact",
    ("drug", "bacteria"): "impact",
    ("food", "bacteria"): "impact",
    ("dietary supplement", "microbiome"): "impact",
    ("drug", "microbiome"): "impact",
    ("food", "microbiome"): "impact",
    ("chemical", "DDF"): "influence",
    ("dietary supplement", "DDF"): "influence",
    ("food", "DDF"): "influence",
    ("chemical", "gene"): "change expression",
    ("dietary supplement", "gene"): "change expression",
    ("drug", "gene"): "change expression",
    ("food", "gene"): "change expression",
    ("chemical", "human"): "administered",
    ("dietary supplement", "human"): "administered",
    ("drug", "human"): "administered",
    ("food", "human"): "administered",
    ("chemical", "animal"): "administered",
    ("dietary supplement", "animal"): "administered",
    ("drug", "animal"): "administered",
    ("food", "animal"): "administered",
    ("DDF", "anatomical location"): "strike",
    ("DDF", "bacteria"): "change abundance",
    ("DDF", "microbiome"): "change abundance",
    ("DDF", "chemical"): "interact",
    ("DDF", "DDF"): "affect",
    ("DDF", "human"): "target",
    ("DDF", "animal"): "target",
    ("drug", "chemical"): "interact",
    ("drug", "drug"): "interact",
    ("drug", "DDF"): "change effect",
    ("human", "biomedical technique"): "used by",
    ("animal", "biomedical technique"): "used by",
    ("microbiome", "biomedical technique"): "used by",
    ("microbiome", "anatomical location"): "located in",
    ("microbiome", "human"): "located in",
    ("microbiome", "animal"): "located in",
    ("microbiome", "gene"): "change expression",
    ("microbiome", "DDF"): "is linked to",
    ("microbiome", "microbiome"): "compared to"
}

# 2. Load the dataset.
with open('processed_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. Build a clean output array, naming it 'relation' in output.
output = []
for item in data:
    relation = valid_relations.get((item["subject_label"], item["object_label"]), "NONE")
    output.append({
        "sample": item["sample"],
        "subject": item["subject"],
        "subject_label": item["subject_label"],
        "object": item["object"],
        "object_label": item["object_label"],
        "relation": relation,
        "relative_subject_start": item["relative_subject_start"],
        "relative_subject_end": item["relative_subject_end"],
        "relative_object_start": item["relative_object_start"],
        "relative_object_end": item["relative_object_end"],
        "doc_id": item["doc_id"]
    })


# 4. Save the result as a nicely formatted JSON file.
with open('processed_test_with_relations.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
