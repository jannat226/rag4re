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
                doc_id=str(idx),
                # "entities": item["entities"],
                metadata = { "relations":item["relations"]}

            )
        )
    print("documents are ",documents)
    return documents

# #Initialize vetcor store client
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="rag4re_store")

embed_model  = SentenceTransformer('all-MiniLM-L6-v2')

#initiate the llm  -> meta-llama/Llama-3.2-3B
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-instruct")
# tokenizer.pad_token = tokenizer.eos_token
# set_global_tokenizer(tokenizer.encode)
# Settings.embed_model = llm
# create the pipeline with Llama embeddings

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



# run the pipeline
train_nodes = pipeline.run(documents=train_documents)
dev_nodes = pipeline.run(documents=dev_documents)

# Mapping from node  to train_items using ref_doc_id
node_to_train_idx = []
for node in train_nodes:
    #node.ref_doc_id gives the doc_id(string) of the source documents
    train_index = int(node.ref_doc_id)
    node_to_train_idx.append(train_index)
    
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
    matrix = np.zeros((len(dev_nodes), len(train_nodes)))  # shape dev_nodes x train_nodes
    for i, dev_node in enumerate(dev_nodes):
        for j, train_node in enumerate(train_nodes):
            score = await evaluate_similarity(dev_node.text, train_node.text)
            matrix[i, j] = score
    return matrix

sim_matrix = asyncio.run(similarity_matrix(dev_nodes,train_nodes))

print("Similarity matrix shape:", sim_matrix.shape)
print(sim_matrix)

#so far embedding and similarity matrix is done 
#Implementing Retrieval 
nearest_indices = np.argmax(sim_matrix, axis = 0)
print(nearest_indices)

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

# # Now create your LLM without needing to access tokenizer later
# llm = HuggingFaceLLM(
#     model_name="meta-llama/Llama-3.2-3B-instruct",
#     device_map="auto",
#     model_kwargs={"torch_dtype": torch.float16, "quantization_config": quantization_config},
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.0, "do_sample": False},
# )
generation_model = Ollama(
    model="llama3.1:latest",
    request_timeout=300,
    context_window=8000,
)




# Extract relation 
relation_types = list(set(valid_relations.values()))
outputs = []

#dev items and predict relation types
for idx, dev_item in enumerate(dev_items):
    query_text = dev_item["metadata"]["abstract"]
   
    retrieved_train_idx = nearest_indices[idx]
    train_item_idx = node_to_train_idx[retrieved_train_idx]
    
    if 0 <= train_item_idx < len(train_items):
        retrieved_text = train_items[train_item_idx]["metadata"]["abstract"]
        example_relations = train_items[train_item_idx].get('relations', [])
        
        if example_relations:
            example_head = example_relations[0].get('subject_text_span', 'N/A')
            example_tail = example_relations[0].get('object_text_span', 'N/A')
            example_relation = example_relations[0].get('predicate', 'N/A')
            # example_relation = valid_relations.get(
            #     (example_relations[0].get('subject_label', ''), example_relations[0].get('object_label', '')), 
            #     'related_to'
            # )
        else:
            example_head, example_tail, example_relation = "N/A", "N/A", "N/A"
    else:
        print(f"Warning: retrieved index {retrieved_train_idx} out of train_items range")
        continue

    relations = dev_item.get('relations', [])
    if not relations:
        continue

    for relation in relations:
        head_entity = relation.get('subject_text_span', 'N/A')
        tail_entity = relation.get('object_text_span', 'N/A')
        
        # Create the messages
        messages = [
             ChatMessage(
            
            role= "system",
            content= f"Find the relationship between the entities, given the most relevant example , the entities in them and their relation.Respond only with a valid JSON. Choose only one relation from this list: [{', '.join(relation_types)}]. Your response MUST be in the form: {{\"relation\": \"<relation_type>\"}}"        
        ),
        ChatMessage(role= "user",
            content= 
                f"Relevant example: {retrieved_text}\n"
                f"Example entities: {example_head}, {example_tail}\n"
                f"Example relation: {example_relation}\n\n"
                f"New sentence: {query_text}\n"
                f"Entities: {head_entity}, {tail_entity}"
            )
        ]
        print("message is ", messages)
       
        
         # Generate prediction
        pred_text = generation_model.chat(messages)
    

        prediction_raw  = pred_text.message.content 
        print("prediction_raw",prediction_raw)
            

        
        def normalize_prediction(pred_text):
            """Normalize prediction to match ground truth format"""
            if not pred_text:
                return "unknown"
            # Convert to lowercase and replace underscores with spaces
            return pred_text.lower().replace('_', ' ').strip()
        
        # Parse the JSON response
        prediction = "unknown"  # Default fallback
        try:
            # Try to parse as JSON first
            prediction_json = json.loads(prediction_raw)
            raw_relation = prediction_json.get("relation", "")
            prediction = normalize_prediction(raw_relation)
            # print(f"From JSON: '{raw_relation}' -> '{prediction}'")
            
        except json.JSONDecodeError:
            print("JSON parsing failed, trying fallback methods...")
            
            # Fallback 1: Extract from raw text
            prediction_text = prediction_raw.split('\n')[0].strip()
            
            # Fallback 2: Try regex extraction
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
            # Keep the prediction anyway for debugging, or set to default:
            # prediction = "related to"  # uncomment if you want a default
        
        outputs.append({
            "messages": messages,
            "prediction": prediction,
            "head": head_entity,
            "tail": tail_entity,
            "ground_truth": relation.get('predicate', 'unknown').lower().strip()
        })
        
        print(f"Final prediction: '{prediction}'")
        print(f"ground_truth: ,{relation.get('predicate', 'unknown').lower().strip()}")
        print("-" * 50)
        
# Save predictions
with open('rag4re_predictions_llama-withValidRelation.json', 'w') as out_f:
    json.dump(outputs, out_f, indent=2)

#evaluation 
wandb.init(project="relation-extraction", name="RAG_flanT5_eval")


print("=== EVALUATION  ===")

all_predictions_dict = {}
all_groundtruths_dict = {}
all_groundtruths = []
all_predictions = []

prediction_index = 0

for dev_item in dev_items:
    relations = dev_item.get('relations', [])
    if not relations:
        continue

    for relation in relations:
        true_relation = relation.get('predicate', 'unknown').lower().strip()
        all_groundtruths.append(true_relation)
        # Count ground truth
        if true_relation in all_groundtruths_dict:
            all_groundtruths_dict[true_relation] += 1
        else:
            all_groundtruths_dict[true_relation] = 1

        # Get corresponding prediction
        if prediction_index < len(outputs):
            pred_rel = outputs[prediction_index]["prediction"]
        else:
            pred_rel = "unknown"
        all_predictions.append(pred_rel)

        # Count prediction
        if pred_rel in all_predictions_dict:
            all_predictions_dict[pred_rel] += 1
        else:
            all_predictions_dict[pred_rel] = 1

        prediction_index += 1

print(f"Arrays: predictions={len(all_predictions_dict)}, ground_truth={len(all_groundtruths_dict)}")
print(f"Unique ground truths: {all_groundtruths_dict}")
print(f"Unique predictions: {all_predictions_dict}")

# Show first few comparisons
print("\nFirst 10 comparisons:")
# for i in range(min(10, len(all_predictions))):
#     p, g = all_predictions[i], all_groundtruths[i]
#     match = "✓" if p == g else "✗"
#     print(f"  {match} pred='{p}' | true='{g}'")

# Calculate metrics
if len(all_predictions) == len(all_groundtruths):
    accuracy = accuracy_score(all_groundtruths, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_groundtruths, all_predictions, average='weighted')
    
    print(f"\nEvaluation RESULTS for one shot:")
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
    print("ERROR!")
# Build rows for the table
rows = []
for idx, out in enumerate(outputs):
    # Each output is tied to a dev_item's abstract (query_text in your loop)
    # We'll add it here explicitly
    sentence = out["messages"][1].content.split("New sentence:")[-1].split("\n")[0].strip()
    
    rows.append({
        "sentence": sentence,
        "entity1": out["head"],
        "entity2": out["tail"],
        "prediction": out["prediction"],
        "ground_truth": out["ground_truth"]
    })

# Create DataFrame
df = pd.DataFrame(rows, columns=["sentence", "entity1", "entity2", "prediction", "ground_truth"])

# Save to CSV for later inspection
df.to_csv("relation_extraction_results.csv", index=False)