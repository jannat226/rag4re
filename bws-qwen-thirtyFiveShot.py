from utils import read_json, write_json
import torch
import json
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import numpy as np
import asyncio
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import qdrant_client
import pandas as pd
import re
from pydantic import BaseModel
import pandas as pd

device = "cuda:0"

class RelationWithReasoning(BaseModel):
    relation: str
    reasoning: str


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

# Load data
processed_dev_file = '/home/lnuj3/thesis/processed_test.json'
dev_items = read_json(processed_dev_file)[:82]
processed_train_file = '/home/lnuj3/thesis/processed_train.json'
train_items = read_json(processed_train_file)[:120]

print(f"Training items: {len(train_items)}")
print(f"Dev items: {len(dev_items)}")

# Prepare documents
def prepare_documents(items):
    documents = []
    for idx, item in enumerate(items):
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

train_documents = prepare_documents(train_items)
dev_documents = prepare_documents(dev_items)

# Initialize vector store with in-memory client
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="rag4re_store")

# Pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=4000, chunk_overlap=50),
        HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2"),
    ],
    vector_store=vector_store,
)

# Run pipeline to get nodes with embeddings
train_nodes = pipeline.run(documents=train_documents)  # This inserts/vectorizes
dev_nodes = pipeline.run(documents=dev_documents)

# Storage context and index setup for retrieval
local_embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.embed_model = local_embed_model
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=train_nodes, storage_context=storage_context, embed_model=local_embed_model)

# Initialize Ollama LLM
generation_model = Ollama(
    model="qwen3:14b",
    request_timeout=300,
    context_window=8000,
)

relation_types = list(set(valid_relations.values()))
outputs = []

# For each dev item:
for idx, dev_item in enumerate(dev_items):
    query_text = dev_item["sample"]
    head_entity = dev_item["subject"]
    tail_entity = dev_item["object"]

    # Retrieve top 35 similar train nodes for few-shot examples
    retriever = index.as_retriever(similarity_top_k=35)
    retrieved_nodes = retriever.retrieve(query_text)

    # Format few-shot examples for prompt
    few_shot_examples = []
    for i, node in enumerate(retrieved_nodes):
        doc_meta = node.node.metadata
        rel = valid_relations.get((doc_meta.get("subject_label"), doc_meta.get("object_label")), "none")
        example_text = (
            f"Example {i+1}:\n"
            f"Sentence: {node.node.text}\n"
            f"Entities: {doc_meta.get('subject')}, {doc_meta.get('object')}\n"
            f"Relation: {rel}\n"
        )
        few_shot_examples.append(example_text)

    few_shot_prompt = "\n---\n".join(few_shot_examples)

    # Compose full user prompt with retrieved shots + new query
    user_prompt = (
        f"{few_shot_prompt}\n---\n"
        f"New sentence: {query_text}\n"
        f"Entities: {head_entity}, {tail_entity}\n"
        "Respond only with a valid JSON in the form {\"relation\": \"<relation>\", \"reasoning\": \"<explanation>\"}."
    )

    messages = [
        ChatMessage(
            role="system",
            content=(
                f"You are an expert relation extractor. Choose one relation from this list: "
                f"[{', '.join(relation_types)}]. "
                "Please provide the relation and reasoning in a JSON object."
            )
        ),
        ChatMessage(role="user", content=user_prompt),
    ]

    pred_text = generation_model.chat(messages, format=RelationWithReasoning.model_json_schema())

    try:
        prediction_obj = RelationWithReasoning.model_validate_json(pred_text.message.content)
        relation = prediction_obj.relation
        reasoning = prediction_obj.reasoning
    except Exception as e:
        print(f"Failed to parse structured output for dev item {idx + 1}: {e}")
        relation = "unknown"
        reasoning = ""

    outputs.append({
        "head": head_entity,
        "tail": tail_entity,
        "subject_label": dev_item["subject_label"],
        "object_label": dev_item["object_label"],
        "prediction": relation,
        "reasoning": reasoning,
        "ground_prediction": valid_relations.get((dev_item["subject_label"], dev_item["object_label"]), 'None')
    })

    print(f"Dev item {idx + 1} - Relation: {relation}")
    print(f"Reasoning: {reasoning}\n---\n")

# Save predictions to JSON file
with open('rag4re_predictions_35shot_rag.json', 'w') as out_f:
    json.dump(outputs, out_f, indent=2)

# Evaluation
wandb.init(project="relation-extraction", name="RAG4RE_35shot_RAG")

all_predictions = [o["prediction"] for o in outputs]
all_groundtruths = [
    valid_relations.get((item["subject_label"], item["object_label"]), 'related_to').lower() for item in dev_items
]

accuracy = accuracy_score(all_groundtruths, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_groundtruths, all_predictions, average='weighted')

print(f"\nEvaluation RESULTS for 35-shot + RAG prompting:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

wandb.log({
    "eval/accuracy": accuracy,
    "eval/precision": precision,
    "eval/recall": recall,
    "eval/f1_weighted": f1
})
matches = sum(1 for p, g in zip(all_predictions, all_groundtruths) if p == g)
total = len(all_predictions)
print(f"Exact matches: {matches} out of {total}")

wandb.finish()

results_table = []

for idx, (dev_item, output) in enumerate(zip(dev_items, outputs)):
    results_table.append({
        "Doc ID": dev_item.get("doc_id", idx),
        "Query": dev_item.get("sample", ""),
        "Entity1": output["head"],
        "Entity2": output["tail"],
        "Predicate (Prediction)": output.get("prediction", ""),
        "Reasoning": output.get("reasoning", ""),
        "Ground Truth": output.get("ground_prediction", "")
    })

df = pd.DataFrame(results_table)
excel_filename = 'relation_extraction_results_35shot_rag.xlsx'
df.to_excel(excel_filename, index=False)

print(f"Saved detailed results to {excel_filename}")
