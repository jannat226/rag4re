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
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import qdrant_client
import pandas as pd
import re
from pydantic import BaseModel
import pandas as pd
import argparse

device = "cuda:0"
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Relation Extraction with RAG")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to dev JSON file")
    parser.add_argument("--num_shots", type=int, default=10, help="Number of few-shot examples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processed_train_file = args.train_file
    processed_dev_file = args.dev_file
    num_shots = args.num_shots
    #checkpoints
    checkpoint_path = 'rag4re_predictions_nonReasoning_Qwen_checkpoint3.json'
    outputs = []
    done_indices = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as ckpt_f:
            outputs = json.load(ckpt_f)
            done_indices = {o.get('dev_idx', i) for i, o in enumerate(outputs)}
    else:
        outputs = []

    class RelationNoReasoning(BaseModel):
        relation: str


    # valid_relations dictionary
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

    # load data
    train_items = read_json(processed_train_file)
    dev_items = read_json(processed_dev_file)

    print(f"Training items: {len(train_items)}")
    print(f"Dev items: {len(dev_items)}")

    # prepare documents
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

    # initialize vector store with in-memory client
    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="rag4re_store")

    # llamaIndex Pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=4000, chunk_overlap=50),
            HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2"),
        ],
        vector_store=vector_store,
    )

    # run pipeline to get nodes with embeddings
    train_nodes = pipeline.run(documents=train_documents) 
    dev_nodes = pipeline.run(documents=dev_documents)

    # storage context and index setup for retrieval
    local_embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    Settings.embed_model = local_embed_model
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=train_nodes, storage_context=storage_context, embed_model=local_embed_model)

    # initialize Ollama LLM
    generation_model = Ollama(
        model="qwen3:14b",
        request_timeout=300,
        context_window=8000,
    )

    relation_types = list(set(valid_relations.values()))
    match_count = 0
    # for each dev item:
    for idx, dev_item in enumerate(dev_items):
        if idx in done_indices:
            continue 
        query_text = dev_item["sample"]
        head_entity = dev_item["subject"]
        tail_entity = dev_item["object"]

        # Retrieve top num_shots similar train nodes for few-shot examples
        retriever = index.as_retriever(similarity_top_k=num_shots)
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

        # full user prompt with retrieved shots + query
        
        user_prompt = (
            f"{few_shot_prompt}\n---\n"
            f"New sentence: {query_text}\n"
            f"Entities: {head_entity}, {tail_entity}\n"
            "Respond only with a valid JSON in the form {\"relation\": \"<relation>\"}.\n"
            "Do not include any reasoning or explanation in the response."
        )


        messages = [
            ChatMessage(
                role="system",
                content=(
                    f"You are an expert relation extractor. Choose one relation from this list: "
                    f"[{', '.join(relation_types)}]. "
                    "Please provide the relation in a JSON object. /no_think"
                )
            ),

            ChatMessage(
                role="user",
                content="/nothink " + user_prompt
            )
            ,
        ]
        
       
        pred_text = generation_model.chat(
            messages,
            format=RelationNoReasoning.model_json_schema(),
            think=False
        )

       
            
        print("Model raw output:")
        print(pred_text.message.content)

        # Then parse as usual if needed
        try:
            prediction_obj = RelationNoReasoning.model_validate_json(pred_text.message.content)
            relation = prediction_obj.relation
            reasoning = ""  # Reasoning expected to be empty in non-reasoning mode

        except Exception as e:
            print(f"Failed to parse structured output: {e}")
            relation = "unknown"
            reasoning = ""

        print("Parsed relation:", relation)
        print("Reasoning (should be empty or minimal):", reasoning)
        outputs.append({
            "dev_idx": idx,
            "head": head_entity,
            "tail": tail_entity,
            "subject_label": dev_item["subject_label"],
            "object_label": dev_item["object_label"],
            "prediction": relation,
            "reasoning": reasoning,
            "ground_prediction": valid_relations.get((dev_item["subject_label"], dev_item["object_label"]), 'None')
        })
        if (len(outputs) % 100 == 0) or (idx == len(dev_items) - 1):
            with open(checkpoint_path, 'w') as ckpt_f:
                json.dump(outputs, ckpt_f, indent=2)
    


        print(f"Dev item {idx + 1} - Relation: {relation}")
        print(f"Reasoning: {reasoning}\n---\n")
        print(
            f"head: {head_entity}\n"
            f"tail: {tail_entity}\n"
            f"subject_label: {dev_item['subject_label']}\n"
            f"object_label: {dev_item['object_label']}\n"
            f"prediction: {relation}\n"
        
            f"ground_prediction: {valid_relations.get((dev_item['subject_label'], dev_item['object_label']), 'None')}\n"
        )
        ground_truth = valid_relations.get((dev_item["subject_label"], dev_item["object_label"]), 'None')
        if relation == ground_truth:
            match_count += 1
        print("the number of match are", match_count)

    # Save predictions to JSON file
    with open(f'rag4re_predictions_{num_shots}shot_rag_midSizeData-nonReasoning_Qwen_.json', 'w') as out_f:
        json.dump(outputs, out_f, indent=2)

    # Evaluation
    wandb.init(project="relation-extraction", name="RAG4RE_{num_shots}shot_RAG_completeData_nonReasoning_Qwen")

    all_predictions = [o["prediction"] for o in outputs]
    all_groundtruths = [
        valid_relations.get((item["subject_label"], item["object_label"]), 'related_to').lower() for item in dev_items
    ]

    accuracy = accuracy_score(all_groundtruths, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_groundtruths, all_predictions, average='weighted')

    print(f"\nEvaluation RESULTS for {num_shots}-shot + RAG prompting:")
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
    excel_filename = f'relation_extraction_results_{num_shots}shot_rag_midSizeData_nonReasoning_Qwen_.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f"Saved detailed results to {excel_filename}")
