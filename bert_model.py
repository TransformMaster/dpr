import json
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import re

model = SentenceTransformer('msmarco-bert-base-dot-v5')
device = torch.device('cuda')
model.to(device)
print(model.device)

path = "./small_dev/docs00.json"
ids = []
vectors = []
with open(path) as f_in:
    for line in f_in:
        info = json.loads(line)
        docid = info['id']
        vector = info['contents']
        vectors.append(vector)
        ids.append(docid)

path = "./collection_jsonl/docs00.json"
pids = []
passages = []
with open(path) as f_in:
    for line in f_in:
        info = json.loads(line)
        docid = info['id']
        vector = info['contents']
        passages.append(vector)
        pids.append(docid)

query_embedding = model.encode(vectors, device=device, show_progress_bar=True)
passage_embedding = model.encode(passages, device=device, show_progress_bar=True)

with open('query_embedding_1.pickle', 'wb') as pkl:
    pickle.dump({'ids': ids, 'embeddings': query_embedding}, pkl, protocol=pickle.HIGHEST_PROTOCOL)

with open('passage_embedding_1.pickle', 'wb') as pkl:
    pickle.dump({'ids': pids, 'embeddings': passage_embedding}, pkl, protocol=pickle.HIGHEST_PROTOCOL)
