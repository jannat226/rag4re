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

dev_items = [item for item in dev_data.values() if 'metadata' in item and 'abstract' in item['metadata']]
# Initialize tokenizer 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
generation_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",device_map = device)

# tokenizer.pad_token = tokenizer.eos_token 
tokenizer.add_special_tokens ({"pad_token": "[PAD]"})
# generation_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xl')

# # Prompt creation
# def create_prompt(query_sentence, retrieved_sentence, head_ent, tail_ent, rel_types, example_head, example_tail, example_relation):
#     relation_list = ", ".join(rel_types)
#     prompt = f"""
# Respond only with a valid JSON. Choose only one relation from this list: [{relation_list}].
# Your response MUST be in the form: {{"relation": "<relation_type>"}}

# Relevant example: {retrieved_sentence}
# Example entities: {example_head}, {example_tail}
# Example relation: {example_relation}

# New sentence: {query_sentence}
# Entities: {head_ent}, {tail_ent}
# """
#     return prompt.strip()



# Extract relation 
relation_types = list(set(valid_relations.values()))
outputs = []

#dev items and predict relation types
for idx, dev_item in enumerate(dev_items):
    query_text = dev_item["metadata"]["abstract"]

    relations = dev_item.get('relations', [])
    if not relations:
        continue

    for relation in relations:
        head_entity = relation.get('subject_text_span', 'N/A')
        tail_entity = relation.get('object_text_span', 'N/A')
        
        # Create the messages
        messages = [
            {
                "role": "system",
                "content": f"Respond only with a valid JSON. Choose only one relation from this list: [{', '.join(relation_types)}]. Your response MUST be in the form: {{\"relation\": \"<relation_type>\"}}"
            },
            {
                "role": "user",
                "content": (
                    f"Sentence: {query_text}\n"
                    f"Entities: {head_entity}, {tail_entity}"
                )
            }
        ]
        
        # Apply chat template and tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_attention_mask=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():  # Save memory
            outputs_ids = generation_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=40,
                do_sample=False,  # For deterministic results
                pad_token_id=tokenizer.pad_token_id
            )
        
        # FIXED: Decode only the generated tokens
        input_length = inputs['input_ids'].shape[-1]
        generated_tokens = outputs_ids[0][input_length:]
        prediction_raw = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"Raw prediction: '{prediction_raw}'")
        
        def normalize_prediction(pred_text):
            """Normalize prediction to match ground truth format"""
            if not pred_text:
                return "unknown"
            # Convert to lowercase and replace underscores with spaces
            return pred_text.lower().replace('_', ' ').strip()
        
        prediction = "unknown"  # Default fallback
        try:
           
            prediction_json = json.loads(prediction_raw)
            raw_relation = prediction_json.get("relation", "")
            prediction = normalize_prediction(raw_relation)
            print(f"From JSON: '{raw_relation}' -> '{prediction}'")
            
        except json.JSONDecodeError:
            print("JSON parsing failed, trying fallback methods...")
            
            # Extract from raw text
            prediction_text = prediction_raw.split('\n')[0].strip()
            
            #regex extraction
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
        
        # Validate prediction is in valid relation types
        if prediction not in [rel.lower() for rel in relation_types]:
            print(f"WARNING: '{prediction}' not in valid relations!")
            print(f"Valid options: {sorted(set([rel.lower() for rel in relation_types]))}")
          
        
        outputs.append({
            "messages": messages,
            "prediction": prediction,
            "head": head_entity,
            "tail": tail_entity,
            "raw_prediction": prediction_raw  # Keep for debugging
        })
        
        print(f"Final prediction: '{prediction}'")
        print("-" * 50)
        
# Save predictions
with open('rag4re_predictions.json', 'w') as out_f:
    json.dump(outputs, out_f, indent=2)

#evaluation 
wandb.init(project="relation-extraction", name="RAG_flanT5_eval")


print("=== EVALUATION ===")

all_predictions = []
all_groundtruths = []

prediction_index = 0

for dev_item in dev_items:
    relations = dev_item.get('relations', [])
    if not relations:
        continue
    
    for relation in relations:
        # FIXED: Use the correct field names from your data structure
        # Your data uses "predicate" not "relation"
        true_relation = relation.get('predicate', 'unknown').lower().strip()
        all_groundtruths.append(true_relation)

        # Get corresponding prediction
        if prediction_index < len(outputs):
            pred_rel = outputs[prediction_index]["prediction"]            
        else:
            pred_rel = "unknown"            
        all_predictions.append(pred_rel)
        prediction_index += 1

print(f"Arrays: predictions={len(all_predictions)}, ground_truth={len(all_groundtruths)}")
print(f"Unique ground truths: {sorted(set(all_groundtruths))}")
print(f"Unique predictions: {sorted(set(all_predictions))}")

# Show first few comparisons
print("\nFirst 10 comparisons:")
for i in range(min(10, len(all_predictions))):
    p, g = all_predictions[i], all_groundtruths[i]
    match = "correct" if p == g else "wrong"
    print(f"  {match} pred='{p}' | true='{g}'")

# Calculate metrics
if len(all_predictions) == len(all_groundtruths):
    accuracy = accuracy_score(all_groundtruths, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_groundtruths, all_predictions, average='weighted')
    
    print(f"\nEvaluation RESULTS for zero shot:")
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
    
    # Calculate matches for manual verification
    matches = sum(1 for p, g in zip(all_predictions, all_groundtruths) if p == g)
    print(f"Exact matches: {matches} out of {len(all_predictions)}")
    
else:
    print("ERROR: Length mismatch between predictions and ground truth!")
    