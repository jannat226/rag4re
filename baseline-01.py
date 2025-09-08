from sentence_transformers import SentenceTransformer
import json 
import numpy as np 



with open('small_train.json') as file :
    train_data = json.load(file)
    
with open('dev5.json') as file :
    dev_data = json.load(file)
    
abstracts = [item['metadata']['abstract'] for ]
#1.Load the pretrained Sentence Transformer model 
model = SentenceTransformer

    
sentences = [item['sample'] for item in train_data] 
query = [item['sample'] for item in dev_data] 
# 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences, show_progress_bar = True)
embeddings = model.encode(sentences)
queries = model.encode(query)

print(queries.shape)
print(embeddings.shape)
# np.save('train_small_embedding')


# 3. Calculate the embedding similarities
similarities = model.similarity(queries, embeddings)
print(similarities)