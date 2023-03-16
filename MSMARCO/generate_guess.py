import json
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import faiss
import numpy as np

def train_faiss_ivf(embeddings, p_idx):
    quantizer = faiss.IndexFlatIP(embeddings
                                  .shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], 1024, faiss.METRIC_INNER_PRODUCT)
    #faiss.normalize_L2(embeddings)
    index.train(embeddings)
    index.add_with_ids(embeddings, p_idx)
    faiss.write_index(index, "centroidpq1024")
    #index = faiss.read_index("centroidpq1024")
    cell_index_list = []
    j = 0
    for (i,k) in zip(embeddings,p_idx):
        _, cell_index = index.quantizer.search(np.array([i]), 1)
        cell_index_list.append([cell_index,k])
        j += 1
        if j%1000 == 0:
            print(j)

    with open('cell_index_list.pickle', 'wb') as fp:
        pickle.dump(cell_index_list, fp)


def train_faiss_exact(embeddings, p_idx):
    #faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, p_idx)
    faiss.write_index(index, "exact")


#with open('./embeddings/passage_embedding_1.pickle', 'rb') as pkl:
#    cache_data = pickle.load(pkl)
#    corpus_sentences = cache_data['ids']
#    corpus_embeddings = cache_data['embeddings']

with open('./embeddings/query_embedding_1.pickle', 'rb') as pkl:
    cache_data1 = pickle.load(pkl)
    query_sentences = cache_data1['ids']
    query_embeddings = cache_data1['embeddings']

#embeddings = np.array([embedding for embedding in corpus_embeddings]).astype("float32")
#p_idx = np.asarray(corpus_sentences).astype(np.int64)
#train_faiss_exact(embeddings, p_idx)
#train_faiss_ivf(embeddings, p_idx)

index = faiss.read_index("centroidpq1024")
cell_index_list = []
query_answers = {}
embeddings1 = np.array([embedding for embedding in query_embeddings]).astype("float32")
index.nprobe = 10
#faiss.normalize_L2(embeddings1)
for i in range(embeddings1.shape[0]):
    vector = embeddings1[i]
    qidx = query_sentences[i]
    D, I = index.search(np.array([vector]), k=10)
    _, cell_index = index.quantizer.search(np.array([vector]), 1)
    cell_index_list.append(cell_index[0])
    query_answers[qidx] = I[0].tolist()
    if i%100 == 0:
        print(i)

with open("ivf102410.tsv", mode='w', encoding="utf-8") as f:
    for (qid, docs), cell in zip(query_answers.items(), cell_index_list):
        ranked = docs
        for i in list(range(len(ranked))):
            f.write('{}\t{}\t{}\t{}\n'.format(qid, ranked[i], i + 1, cell))

#with open("100exact.tsv", mode='w', encoding="utf-8") as f:
#    for qid, docs in query_answers.items():
#        ranked = docs
#        for i in list(range(len(ranked))):
#            f.write('{}\t{}\t{}\n'.format(qid, ranked[i], i + 1))


