from utils import read_json, write_json
import torch
import json
from llama_index.core import Document, set_global_tokenizer, VectorStoreIndex,StorageContext, Settings
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig, AutoTokenizer , AutoModelForSeq2SeqLM
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

# from llama_index.embeddings.openai import OpenAIEmbedding

import qdrant_client
from sentence_transformers import SentenceTransformer


small_dev_data = '/home/lnuj3/thesis/dev5.json'
small_train_data = '/home/lnuj3/thesis/small_train.json'
beforeDoc = '/home/lnuj3/thesis/beforeDoc'
docs = '/home/lnuj3/thesis/docs'
device="cuda:0"
valid_relations = { 
    ("anatomical location", "human"): "located in",
    ("anatomical location", "animal"): "located in",
    ("bacteria", "bacteria"): "interact",
    ("bacteria", "chemical"): "interact",
    ("bacteria", "drug"): "interact",
    ("bacteria", "DDF"): "influence",
    ("bacteria", "gene"): "change expression",
    ("bacteria", "human" ):"located in",
    ("bacteria", "animal" ):"located in",
    ("bacteria", "microbiome" ):"part of",
    ("chemical", "anatomical location" ):"located in",
    ("chemical", "human" ):"located in",
    ("chemical", "animal" ):"located in",
    ("chemical", "chemical"):"interact",
    ("chemical", "chemical"): "part of",
    ("chemical", "microbiome" ):"impact",
    ("chemical", "microbiome" ):"produced by",
    ("chemical", "bacteria" ):"impact",
    ("dietary supplement" ,"bacteria"): "impact",
    ("drug", "bacteria" ):"impact",
    ("food", "bacteria" ):"impact",
    ("chemical", "microbiome" ):"impact",
    ("dietary supplement", "microbiome"): "impact",
    ("drug", "microbiome" ):"impact",
    ("food", "microbiome" ):"impact",
    ("chemical", "DDF" ):"influence",
    ("dietary supplement", "DDF" ):"influence",
    ("food", "DDF" ):"influence",
    ("chemical", "gene" ):"change expression",
    ("dietary supplement", "gene" ):"change expression",
    ("drug", "gene" ):"change expression",
    ("food", "gene" ):"change expression",
    ("chemical", "human" ):"administered",
    ("dietary supplement", "human" ):"administered",
    ("drug", "human" ):"administered",
    ("food", "human" ):"administered",
    ("chemical", "animal" ):"administered",
    ("dietary supplement", "animal" ):"administered",
    ("drug", "animal" ):"administered",
    ("food", "animal" ):"administered",
    ("DDF", "anatomical location" ):"strike",
    ("DDF", "bacteria" ):"change abundance",
    ("DDF", "microbiome" ):"change abundance",
    ("DDF", "chemical" ):"interact",
    ("DDF", "DDF" ):"affect",
    ("DDF", "DDF" ):"is a",
    ("DDF", "human" ):"target",
    ("DDF", "animal" ):"target",
    ("drug", "chemical" ):"interact",
    ("drug", "drug" ):"interact",
    ("drug", "DDF" ):"change effect",
    ("human", "biomedical technique" ):"used by",
    ("animal", "biomedical technique" ):"used by",
    ("microbiome", "biomedical technique" ):"used by",
    ("microbiome", "anatomical location" ):"located in",
    ("microbiome", "human" ):"located in",
    ("microbiome", "animal" ):"located in",
    ("microbiome", "gene" ):"change expression",
    ("microbiome", "DDF" ):"is linked to",
    ("microbiome", "microbiome" ):"compared to"
}

dev_data = read_json(small_dev_data)
train_data = read_json(small_train_data)


dev_items = [item for item in dev_data.values() if 'metadata' in item and 'abstract' in item['metadata']]
train_items = [item for item in train_data.values() if 'metadata' in item and 'abstract' in item['metadata']]
write_json(beforeDoc, train_items)

print("this is dev items ", train_items)
#preparing the documents

def prepare_documents(train_items):
    documents = []
    for idx, item in enumerate(train_items):
        documents.append(
            Document(
                text=item["metadata"]["abstract"],
                doc_id=str(idx),  # Important for reverse mapping
                metadata={"relations": item["relations"]}
            )
        )
    return documents

# #Initialize vetcor store client
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="rag4re_store")

embed_model  = SentenceTransformer('all-MiniLM-L6-v2')

#initiate the llm  -> meta-llama/Llama-3.2-3B
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=4000, chunk_overlap=50),
        # TitleExtractor(llm=llm),  # pass LLM explicitly here
        HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2"),
    ],
    vector_store=vector_store,
)

train_documents = prepare_documents(train_items)
dev_documents =  prepare_documents(dev_items)

# run the pipeline to get node chunks 
train_nodes = pipeline.run(documents=train_documents)
dev_nodes = pipeline.run(documents=dev_documents)

#node to train_item index mapping from ref_doc_id
node_to_train_idx = [int(node.ref_doc_id) for node in train_nodes]


print(f"Created  train {len(train_nodes)} nodes with embeddings")
print(f"Created  dev {len(dev_nodes)} nodes with embeddings")

local_embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")


Settings.embed_model = local_embed_model
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# Storage context from  Qdrant vector store

index = VectorStoreIndex(
    nodes=train_nodes,
    storage_context=storage_context,
    embed_model=local_embed_model,
)

# Semantic similarity evaluator
evaluator = SemanticSimilarityEvaluator()

async def evaluate_similarity(text1, text2):
    result = await evaluator.aevaluate(response=text1, reference=text2)
    return result.score

async def similarity_matrix(dev_nodes, train_nodes):
    matrix = np.zeros((len(dev_nodes), len(train_nodes)))
    for i, dev_node in enumerate(dev_nodes):
        for j, train_node in enumerate(train_nodes):
            matrix[i, j] = await evaluate_similarity(dev_node.text, train_node.text)
    return matrix

sim_matrix = asyncio.run(similarity_matrix(dev_nodes, train_nodes))


print("Similarity matrix shape:", sim_matrix.shape)
print(sim_matrix)


#prepare relation_type
relation_types = list(set(valid_relations.values()))
outputs = []
total_relations = sum(len(item.get('relations', [])) for item in dev_items)
print(f"Total relations: {total_relations}")

count_predictions = 0
for idx, dev_item in enumerate(dev_items):
    relations = dev_item.get('relations', [])
    for relation in relations:
        count_predictions += 1

print(f"Total predictions generated: {count_predictions}")
# Initialize tokenizer 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
generation_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",device_map = device)

# tokenizer.pad_token = tokenizer.eos_token 
tokenizer.add_special_tokens ({"pad_token": "[PAD]"})


# Get top-k indices of train nodes per dev node (descending order)   
top_k = 5
top_k_indices_per_dev = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :top_k]

# Now for each dev item, prepare the prompt examples with mapping node idx -> train item idx
all_outputs = []
def safe_extract_relation(text):
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "relation" in data:
            return data["relation"].lower().replace("_", " ")
    except Exception:
        pass
    
    # Regex fallback
    match = re.search(r'"relation"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1).lower().replace("_", " ")
    
    return "unknown"
for idx, dev_item in enumerate(dev_items):
    query_text = dev_item["metadata"]["abstract"]
    relations = dev_item.get('relations', [])
    if not relations:
        continue

    top_k_indices = top_k_indices_per_dev[idx]
    examples_str = ""

    for i, node_idx in enumerate(top_k_indices):
        if node_idx >= len(node_to_train_idx):
            print(f"Warning: node_idx {node_idx} out of range for node_to_train_idx")
            continue
        train_item_idx = node_to_train_idx[node_idx]
        if train_item_idx >= len(train_items):
            print(f"Warning: train_item_idx {train_item_idx} out of range for train_items")
            continue

        retrieved_text = train_items[train_item_idx]["metadata"]["abstract"]
        example_relations = train_items[train_item_idx].get('relations', [])
        if example_relations:
            example_head = example_relations[0].get('subject_span', 'N/A')
            example_tail = example_relations[0].get('object_span', 'N/A')
            # example_relation = valid_relations.get(
            #     (example_relations[0].get('subject_label', ''), example_relations[0].get('object_label', '')),
            #     'related_to'
            # )
            example_relation = example_relations[0].get('predicate', 'N/A')
        else:
            example_head, example_tail, example_relation = "N/A", "N/A", "related_to"

        examples_str += (
            f"Relevant example {i+1}: {retrieved_text}\n"
            f"Example entities {i+1}: {example_head}, {example_tail}\n"
            f"Example relation {i+1}: {example_relation}\n\n"
        )

    relation_list_str = ", ".join(relation_types)

    for relation in relations:
        head_entity = relation.get('subject_span', 'N/A')
        tail_entity = relation.get('object_span', 'N/A')

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a relation extraction assistant. "
                    f"Respond ONLY with valid JSON of the form: {{\"relation\": \"<one of these>\"}}. "
                    f"The allowed relations are: [{relation_list_str}]. "
                    f"Do NOT output explanations, lists, or multiple answers."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{examples_str}"
                    f"New sentence: {query_text}\n"
                    f"Entities: {head_entity}, {tail_entity}"
                )
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_attention_mask=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generation_output = generation_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_tokens = generation_output[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        


        try:
            prediction_json = json.loads(generated_text)
            pred_relation = prediction_json.get("relation", "unknown")
            pred_relation = pred_relation.lower().replace("_", " ")
        except Exception:

            pred_relation = generated_text.split("\n")[0].lower().replace("_", " ")
        pred_relation = safe_extract_relation(generated_text)
        
        if pred_relation not in [r.lower() for r in valid_relations.values()]:
            print(f"Warning: prediction '{pred_relation}' not in valid relations.")

        all_outputs.append({
            "messages": messages,
            "prediction": pred_relation,
            "head": head_entity,
            "tail": tail_entity,
            "ground_truth": relation.get('predicate', 'unknown').lower().strip()
        })

        print(f"Prediction: {pred_relation}")

with open('rag4re_predictions_baseline_with_llama_for_ten_shot.json', 'w') as out_f:
    json.dump(all_outputs, out_f, indent=2)

#evaluation 
wandb.init(project="relation-extraction", name="RAG_flanT5_eval")


print("=== EVALUATION  ===")

all_predictions = [item["prediction"] for item in all_outputs]
all_groundtruths = [item["ground_truth"] for item in all_outputs]

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
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_groundtruths, all_predictions, average='weighted'
    )
    
    print(f"\nEvaluation RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Log to wandb
    wandb.log({
        "eval/accuracy_fixed": accuracy,
        "eval/precision_fixed": precision,
        "eval/recall_fixed": recall,
        "eval/f1_weighted_fixed": f1
    })
    
    # Exact match count
    matches = sum(1 for p, g in zip(all_predictions, all_groundtruths) if p == g)
    print(f"Exact matches: {matches} out of {len(all_predictions)}")
else:
    print("ERROR: mismatch between prediction and ground truth count!")