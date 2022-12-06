import pickle
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import time

# print(type(data)) # dict
# keys: ids, embeddings.   ids 是一个list，['Texas_annexation', xxx]. embeddings 是一个二维数组
# 里面都是数字 numpy array
# shape: (6818, 768)

# print(sentence_embeddings.shape) # (6818,768)
# print(len(idx)) # 6818

# print("idx: \t", idx)
# print("p_idx: \t", p_idx)
# print(len(p_idx)) # 6818



def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


if __name__ == '__main__':
    file = open('./DPR_passage_embedding_2.pickle', 'rb')
    data = pickle.load(file)
    file.close()

    sentence_embeddings = data["embeddings"]
    idx = data["ids"]

    count = 1
    iddict = {}
    p_idx = []
    for id in idx:
        if id not in iddict.keys():
            count += 1
            iddict[id] = count
        p_idx.append(count)

    d = data["embeddings"].shape[1]
    index = faiss.IndexFlatL2(d)
    index2 = faiss.IndexIDMap(index)
    index2.add_with_ids(sentence_embeddings, np.array(p_idx))

    for k in range(1, 100):
        start_time = time.time()

        xq = sentence_embeddings
        D, I = index2.search(xq, k)

        cnt = 0
        for kn in I:
            a = False
            temp = idx[kn[0]]
            for k in kn[1:]:
                if idx[k] == temp:
                    a = True
                # print(idx[k])
            if a: cnt += 1

        end_time = time.time()
        print("final accuracy: \t", cnt/len(p_idx))

        f = open('./NEW_Accuracy_no_ivf_FAISS_DPR.txt', 'a')
        f.write('%s \n' % str(cnt/len(p_idx)))
        f.close()

        f = open('./NEW_Time_no_ivf_FAISS_DPR.txt', 'a')
        f.write('%s \n' % str(end_time - start_time))
        f.close()
