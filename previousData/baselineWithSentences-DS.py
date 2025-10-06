# Corrected RAG4RE for new data structure
from utils import read_json, write_json
import torch
import json
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import asyncio
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from llama_index.llms.ollama import Ollama
import qdrant_client

device = "cuda:0"

# Same valid_relations dictionary
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
    ("chemical", "chemical"): "part of",
    ("chemical", "microbiome"): "impact",
    ("chemical", "microbiome"): "produced by",
    ("chemical", "bacteria"): "impact",
    ("dietary supplement", "bacteria"): "impact",
    ("drug", "bacteria"): "impact",
    ("food", "bacteria"): "impact",
    ("chemical", "microbiome"): "impact",
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
    ("DDF", "DDF"): "is a",
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

print("Loading processed data...")
processed_dev_file = '/home/lnuj3/thesis/processed_test.json'
dev_items = read_json(processed_dev_file)

dev_items = dev_items[:82]

processed_train_file = '/home/lnuj3/thesis/processed_train.json'
train_items = read_json(processed_train_file)

train_items = train_items[:120]



print(f"Training items: {len(train_items)}")
print(f"Dev items: {len(dev_items)}")


def prepare_documents(items):
    print("Preparing documents...")
    documents = []
    for idx, item in enumerate(items) :
        documents.append(
            Document(
                text=item["sample"],  
                doc_id=str(idx),
                metadata={
                    "subject": item["subject"],
                    "subject_label": item["subject_label"],
                    "object": item["object"],
                    "object_label": item["object_label"],
                    "doc_id": item.get("doc_id", "")
                }
            )
        )
    return documents

# Initialize vector store
print("Initializing vector store...")
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="rag4re_store")

# Create pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=4000, chunk_overlap=50),
        HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2"),
    ],
    vector_store=vector_store,
)

# Prepare documents
train_documents = prepare_documents(train_items)
dev_documents = prepare_documents(dev_items)

# Run pipeline
print("Running pipeline...")
train_nodes = pipeline.run(documents=train_documents)
dev_nodes = pipeline.run(documents=dev_documents)

print(f"Created {len(train_nodes)} train nodes with embeddings")
print(f"Created {len(dev_nodes)} dev nodes with embeddings")

# Set up settings
local_embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = local_embed_model
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=train_nodes, storage_context=storage_context, embed_model=local_embed_model)

# Semantic similarity evaluation
print("Computing similarity matrix...")
evaluator = SemanticSimilarityEvaluator()

async def evaluate_similarity(text1, text2):
    result = await evaluator.aevaluate(response=text1, reference=text2)
    return result.score

async def similarity_matrix(dev_nodes, train_nodes):
    matrix = np.zeros((len(dev_nodes), len(train_nodes)))
    for i, dev_node in enumerate(dev_nodes):
        if i % 10 == 0:
            print(f"Processing similarity for dev item {i+1}/{len(dev_nodes)}")
        for j, train_node in enumerate(train_nodes):
            score = await evaluate_similarity(dev_node.text, train_node.text)
            matrix[i, j] = score
    return matrix

# Calculate similarity matrix 
sim_matrix = asyncio.run(similarity_matrix(dev_nodes, train_nodes))
print("Similarity matrix shape:", sim_matrix.shape)

# Get nearest indices (FIXED: axis=1 for dev x train matrix)
nearest_indices = np.argmax(sim_matrix, axis=1)
print(f"Nearest indices: {nearest_indices}")

# Initialize tokenizer and model
print("Loading tokenizer and model...")


tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
generation_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True,  torch_dtype=torch.float16 )
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Prepare relation types
relation_types = list(set(valid_relations.values()))
outputs = []

print(f"Starting prediction generation for {len(dev_items)} dev items...")


for idx, dev_item in enumerate(dev_items):
    query_text = dev_item["sample"] 
    retrieved_train_idx = nearest_indices[idx]

    if 0 <= retrieved_train_idx < len(train_items):

        retrieved_item = train_items[retrieved_train_idx]
        retrieved_text = retrieved_item["sample"]
        
        subject_label = retrieved_item["subject_label"]
        object_label = retrieved_item["object_label"]
        example_relation = retrieved_item["relation"]
        example_relation = valid_relations.get((subject_label, object_label), 'None')
        example_head = retrieved_item["subject"]
        example_tail = retrieved_item["object"]
    else:
        print(f"Warning: retrieved index {retrieved_train_idx} out of range")
        continue

    head_entity = dev_item["subject"]
    tail_entity = dev_item["object"]
    
    # Create messages
    messages = [
        {
            "role": "system",
            "content": f"Find the relationship between the entities, given the most relevant example , the entities in them and their relation.Respond only with a valid JSON. Choose only one relation from this list: [{', '.join(relation_types)}]. Your response MUST be in the form: {{\"relation\": \"<relation_type>\"}}"
        },
        {
            "role": "user",
            "content": (
                f"Relevant example: {retrieved_text}\n"
                f"Example entities: {example_head}, {example_tail}\n"
                f"Example relation: {example_relation}\n\n"
                f"New sentence: {query_text}\n"
                f"Entities: {head_entity}, {tail_entity}"
            )
        }
    ]
    
    # Generate prediction
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    print("message is ", messages)
    # inputs = tokenizer.apply_chat_template(
	# messages,
	# add_generation_prompt=True,
	# tokenize=True,
	# return_dict=True,
	# return_tensors="pt",
    # ).to(generation_model.device)
    
    with torch.no_grad():
        outputs_ids = generation_model.generate(
            input_ids=inputs['input_ids'],
            # attention_mask=inputs['attention_mask'],
            max_new_tokens=40,
            pad_token_id=tokenizer.pad_token_id
        )
   
    # Decode prediction
    input_length = inputs['input_ids'].shape[-1]
    generated_tokens = outputs_ids[0][input_length:]
    prediction_raw = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def normalize_prediction(pred_text):
        if not pred_text:
            return "unknown"
        return pred_text.lower().replace('_', ' ').strip()
    

    prediction = "unknown"
    
    try:
        prediction_json = json.loads(prediction_raw)
        raw_relation = prediction_json.get("relation", "")
        prediction = normalize_prediction(raw_relation)
        # print(f"From JSON: '{raw_relation}' -> '{prediction}'")
    except json.JSONDecodeError:
        print("JSON parsing failed, trying fallback methods...")
        prediction_text = prediction_raw.split('\n')[0].strip()
        if '{"relation":' in prediction_text:
            import re
            match = re.search(r'"relation":\s*"([^"]*)"', prediction_text)
            if match:
                raw_relation = match.group(1)
                prediction = normalize_prediction(raw_relation)
                print(f"From regex: '{raw_relation}' -> '{prediction}'")
            else:
                prediction = normalize_prediction(prediction_text)
        else:
            prediction = normalize_prediction(prediction_text)
    
    outputs.append({
        "prediction": prediction,
        "head": head_entity,
        "tail": tail_entity,
        "subject_label": dev_item["subject_label"],
        "object_label": dev_item["object_label"],
        "raw_prediction": prediction_raw,
        # "ground_prediction": dev_item["relation"]
        "ground_prediction": valid_relations.get((subject_label, object_label), 'None')
    })
   
    print(f"Final prediction: '{prediction}'")

# Save predictions
print("Saving predictions...")
with open('rag4re_predictions_sentence.json', 'w') as out_f:
    json.dump(outputs, out_f, indent=2)

# Evaluation using subject/object labels
print("Starting evaluation...")
wandb.init(project="relation-extraction", name="RAG4RE_new_format")

all_predictions = []
all_groundtruths = []

for i, (dev_item, output) in enumerate(zip(dev_items, outputs)):
    # Generate ground truth using subject_label and object_label
    subject_label = dev_item["subject_label"]
    object_label = dev_item["object_label"]
    # true_relation = dev_item["relation"]
    true_relation = valid_relations.get((subject_label, object_label), 'related_to').lower()
    all_groundtruths.append(true_relation)
    all_predictions.append(output["prediction"])

print(f"Arrays: predictions={len(all_predictions)}, ground_truth={len(all_groundtruths)}")
print(f"Unique ground truths: {sorted(set(all_groundtruths))}")
print(f"Unique predictions: {sorted(set(all_predictions))}")

# Show first few comparisons
print("\nFirst 10 comparisons:")
for i in range(min(10, len(all_predictions))):
    p, g = all_predictions[i], all_groundtruths[i]
    match = "✓" if p == g else "✗"
    print(f"  {match} pred='{p}' | true='{g}'")

# Calculate metrics
if len(all_predictions) == len(all_groundtruths):
    accuracy = accuracy_score(all_groundtruths, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_groundtruths, all_predictions, average='weighted')
    
    print(f"\n Evaluation RESULTS for baseline with sentences:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Log to wandb
    wandb.log({
        "eval/accuracy": accuracy,
        "eval/precision": precision,
        "eval/recall": recall,
        "eval/f1_weighted": f1
    })
    
    matches = sum(1 for p, g in zip(all_predictions, all_groundtruths) if p == g)
    print(f"Exact matches: {matches} out of {len(all_predictions)}")
    
else:
    print("ERROR: Length mismatch between predictions and ground truth!")

wandb.finish()
