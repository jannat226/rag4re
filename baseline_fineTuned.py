from utils import read_json, write_json
import torch
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
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


# Load your JSON file containing training items
with open("processed_train.json", "r") as f:
    train_data = json.load(f)

train_examples = train_items

# Then create DataLoader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model = SentenceTransformer('all-MiniLM-L6-v2')
# Define loss function for fine-tuning
train_loss = losses.CosineSimilarityLoss(model=model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

# Save the fine-tuned model
model.save('fine_tuned_model')

# Later, load the fine-tuned model for embeddings
embed_model = SentenceTransformer('fine_tuned_model')
# embed_model  = SentenceTransformer('all-MiniLM-L6-v2')
class CustomSentenceTransformerEmbedding:
    def __init__(self, model):
        self.model = model
    
    def embed(self, texts):
        # texts is a list of strings
        return self.model.encode(texts, convert_to_tensor=True)

embedding = CustomSentenceTransformerEmbedding(embed_model)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=4000, chunk_overlap=50),
        embedding,
    ],
    vector_store=vector_store,
)


#initiate the llm  -> meta-llama/Llama-3.2-3B
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

train_documents = prepare_documents(train_items)
dev_documents =  prepare_documents(dev_items)

# run the pipeline
train_nodes = pipeline.run(documents=train_documents)
dev_nodes = pipeline.run(documents=dev_documents)


print(f"Created  train {len(train_nodes)} nodes with embeddings")
print(f"Created  dev {len(dev_nodes)} nodes with embeddings")



# Semantic similarity evaluator
evaluator = SemanticSimilarityEvaluator()

async def evaluate_similarity(text1, text2):
    result = await evaluator.aevaluate(response=text1, reference=text2)
    return result.score

async def similarity_matrix(train_nodes, dev_nodes):
    matrix = np.zeros((len(train_nodes), len(dev_nodes)))
    for i, train_node in enumerate(train_nodes):
        for j, dev_node in enumerate(dev_nodes):
            score = await evaluate_similarity(train_node.text, dev_node.text)
            matrix[i, j] = score
    return matrix

# async similarity matrix calculation
sim_matrix = asyncio.run(similarity_matrix(train_nodes, dev_nodes))

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

# Initialize tokenizer 
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
generation_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xl')

# Prompt creation
def create_prompt(query_sentence, retrieved_sentence, head_ent, tail_ent, rel_types, example_head, example_tail, example_relation):
    relation_list = ", ".join(rel_types)
    prompt = f"""
Problem Definition: Relation extraction to detect the relationship between two entities in a sentence.

Example:
Sentence: {retrieved_sentence}
Head Entity: {example_head}
Tail Entity: {example_tail}
Relation: {example_relation}

Now, given the query:

Relevant Example Sentence: {retrieved_sentence}

Query Sentence: {query_sentence}

Head: {head_ent}
Tail: {tail_ent}

Relation types: {relation_list}

Output format: relation_type
"""
    return prompt.strip()

# Extract relation 
relation_types = list(set(valid_relations.values()))

outputs = []

#dev items and predict relation types
for idx, dev_item in enumerate(dev_items):
    query_text = dev_item["metadata"]["abstract"]
    retrieved_train_idx = nearest_indices[idx]

    if 0 <= retrieved_train_idx < len(train_items):
        retrieved_text = train_items[retrieved_train_idx]["metadata"]["abstract"]
        example_relations = train_items[retrieved_train_idx].get('relations', [])
        print("example_relations",example_relations)
        if example_relations:
            example_head = example_relations[0].get('subject_text_span', 'N/A')
            example_tail = example_relations[0].get('object_text_span', 'N/A')
            example_relation = valid_relations.get(
                (example_relations[0].get('subject_type', ''), example_relations[0].get('object_type', '')), 
                'related_to'
            )
        else:
            example_head, example_tail, example_relation = "N/A", "N/A", "related_to"
    else:
        print(f"Warning: retrieved index {retrieved_train_idx} out of train_items range but from the embeddings coz nodes size > normal dev set")
        continue

    relations = dev_item.get('relations', [])
    if not relations:
        continue

    for relation in relations:
        head_entity = relation.get('subject_text_span', 'N/A')
        tail_entity = relation.get('object_text_span', 'N/A')

        prompt = create_prompt(query_text, retrieved_text, head_entity, tail_entity, relation_types,
                               example_head, example_tail, example_relation)

        # Tokenize and generate output
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # print("inputs", inputs)
        outputs_ids = generation_model.generate(**inputs, max_length=16)
        print("output ids", outputs_ids)
        prediction = tokenizer.decode(outputs_ids[0], skip_special_tokens=True).strip().lower().replace(" ", "_")

        outputs.append({
            "prompt": prompt,
            "prediction": prediction,
            "head": head_entity,
            "tail": tail_entity
        })

# Save predictions
with open('rag4re_predictions.json', 'w') as out_f:
    json.dump(outputs, out_f, indent=2)

#evaluation 
wandb.init(project="relation-extraction", name="RAG_flanT5_eval")
all_predictions = []
all_groundtruths = []
print(f"Number of predictions: {len(outputs)}")
print(f"Number of total relations in dev_items: {sum(len(item.get('relations', [])) for item in dev_items)}")

prediction_index = 0

for dev_item in dev_items:
    relations = dev_item.get('relations', [])
    if not relations:
        continue
    for relation in relations:
        #Ground truth label
        true_relation = valid_relations.get(
            (relation.get('subject_type', ''), relation.get('object_type', '')),
            'related_to'
        ).lower().replace(" ", "_")
        all_groundtruths.append(true_relation)

        if prediction_index < len(outputs):
            pred_rel = outputs[prediction_index]["prediction"]            
        else:
            #missing prediction case
            pred_rel = "unknown"            
        all_predictions.append(pred_rel)
        prediction_index += 1

assert len(all_predictions) == len(all_groundtruths), "Mismatch in predictions and ground truths length"

# Compute metrics
accuracy = accuracy_score(all_groundtruths, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_groundtruths, all_predictions, average='weighted')

# Log metrics to wandb
wandb.log({
    "eval/accuracy": accuracy,
    "eval/precision": precision,
    "eval/recall": recall,
    "eval/f1_weighted": f1
})

print(f"Evaluation Accuracy: {accuracy}")
print(f"Evaluation Precision: {precision}")
print(f"Evaluation Recall: {recall}")
print(f"Evaluation F1 (weighted): {f1}")

# Finish wandb run if you're done
wandb.finish()