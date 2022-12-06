import pickle
from sentence_transformers import SentenceTransformer, util
import faiss
import time

file = open('passage_embedding_2.pickle', 'rb')

data = pickle.load(file)

file.close()

sentence_embeddings = data["embeddings"]
idx = data["ids"]
print(sentence_embeddings.shape)
print(len(idx))


count = 0
iddict = {}
p_idx = []
for id in idx:
    if id not in iddict.keys():
        count += 1
        iddict[id] = count
    p_idx.append(count)
print(len(p_idx))


d = data["embeddings"].shape[1]
index = faiss.IndexFlatL2(d)

# ivf
index2 = faiss.IndexIVFFlat(index, d, 256)
faiss.normalize_L2(data["embeddings"])
index2.train(data["embeddings"])
index2.add_with_ids(data["embeddings"], p_idx)
faiss.write_index(index2, "1centroid256")
# ivf end

# index2 = faiss.IndexIDMap(index)
# index2.add_with_ids(sentence_embeddings, p_idx)

start_time = time.time()
k = 20
# index2.add(sentence_embeddings)
# print(index.ntotal)

xq = sentence_embeddings
D, I = index2.search(xq, k)
# print(I)
# print(D)

cnt = 0
for kn in I:
    a = False
    temp = idx[kn[0]]
    for k in kn[1:]:
        if idx[k] == temp:
            a = True
        # print(idx[k])
    if a: cnt += 1
print("--- %s seconds ---" % (time.time() - start_time))
print(cnt/len(p_idx))