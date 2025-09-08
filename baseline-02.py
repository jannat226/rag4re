import json
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM

from utils import read_json, write_json

smallDevData = '/home/lnuj3/thesis/dev5.json'
smallTrainData = '/home/lnuj3/thesis/small_train.json'

devData = read_json(smallDevData)
trainData = read_json(smallTrainData)

    
# print(devData["36532064"])
#extract abstract 
devBeforeEmbedding = []
for item in devData.values():
    if 'metadata' in item and 'abstract' in item['metadata']:
        devBeforeEmbedding.append(item['metadata']['abstract'])


        
trainBeforeEmbedding = []

for item in trainData.values():
    if 'metadata' in item and 'abstract' in item['metadata']:
        trainBeforeEmbedding.append(item['metadata']['abstract'])


dev_items = [item for item in devData.values() if 'metadata' in item and 'abstract' in item['metadata']]
   
# print(trainBeforeEmbedding)

#load model
model = SentenceTransformer('all-MiniLM-L6-v2')
devEmbedding = model.encode(devBeforeEmbedding)
trainEmbedding = model.encode(trainBeforeEmbedding)
# print(devEmbedding.shape, trainEmbedding.shape)
    
    
#normailizing the embeddings 
#np.linalg.norm(trainEmbedding, axis=1, keepdims=True) computes the length (norm) of each embedding vector
trainEmbedding_norm = trainEmbedding / np.linalg.norm(trainEmbedding, axis = 1 , keepdims = True)
devEmbedding_norm = devEmbedding / np.linalg.norm(devEmbedding, axis = 1 , keepdims = True)

# Cosine similarity: Each cell (i, j) is similarity between dev i and train j
similarity = np.dot(devEmbedding_norm, trainEmbedding_norm.T)  # (N_dev, N_train)

# For each dev sample, get index of most similar train sample
nearest_indices = np.argmax(similarity, axis=1)
print(nearest_indices)

#Retrieval 

relevant_examples = [trainBeforeEmbedding[idx] for idx in nearest_indices]
print("this is the relevant examples-0", relevant_examples[0])

# #Prompt 
# for i , dev_abs in enumerate(devBeforeEmbedding):
#     prompt = f'''
#     Problem Definition : Relationship extraction  to detect the relationship
#     between two entities in same sentece.
    
#     Relevant example Senteces : {relevant_examples[i]}
#     Query Sentence : {dev_abs}
#     #Add entity, relation types as needed 
#     output format : relation_type'''
#     print(prompt)

#load the model-- for rag
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xl')

def create_prompt (query_sentence, retrieved_sentence, head_ent, tail_ent, rel_types) :
    prompt = f"""
Problem Definition: Relation extraction to detect the relationship between two entities in a sentence.

Relevant Example Sentence: {retrieved_sentence}

Query Sentence: {query_sentence}

Head: {head_ent}
Tail: {tail_ent}

Relation types: {', '.join(rel_types)}

Output format: relation_type
"""
    return prompt.strip()

outputs = []

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

# Extract unique relation type strings for prompt
relation_types = list(set(valid_relations.values()))
# print(relation_types)

outputs = []
for idx, item in enumerate(dev_items):
    query_text = devBeforeEmbedding[idx]
    print(query_text)
    retrieved_text = trainBeforeEmbedding[nearest_indices[idx]]
    print(retrieved_text)

    relations = item.get('relations', [])
    print("this is relations", relations)
    if not relations:
        # If no relations data, skip
        continue

    for relation in relations:
        head_entity = relation.get('subject_text_span', 'N/A')
        tail_entity = relation.get('object_text_span', 'N/A')

        prompt = create_prompt(query_text, retrieved_text, head_entity, tail_entity, relation_types)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs_ids = model.generate(**inputs, max_length=16)
        prediction = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)

        prediction = prediction.lower().replace(" ", "_")

        outputs.append({
            "prompt": prompt,
            "prediction": prediction,
            "head": head_entity,
            "tail": tail_entity
        })

        print(f"Query #{idx + 1} Head: {head_entity}, Tail: {tail_entity} Prediction: {prediction}\n")


# Save predictions to JSON file
with open('rag4re_predictions_without_llama.json', 'w') as out_f:
    json.dump(outputs, out_f, indent=2)
