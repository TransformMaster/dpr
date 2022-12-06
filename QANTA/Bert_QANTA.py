import json
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
from qbdata import QantaDatabase


model = SentenceTransformer('msmarco-bert-base-dot-v5')
device = torch.device('cuda:0')
model.to(device)
print(model.device)

path = "./qantatest.json"

guesstrain = QantaDatabase(path)
questions = [x.text for x in guesstrain.guess_test_questions]
answers = [x.page for x in guesstrain.guess_test_questions]
tokens = [x.tokenizations for x in guesstrain.guess_test_questions]

ids = []
passages = []

for index in range(len(questions)):
    question = questions[index]
    for token in tokens[index]:
        passage = question[token[0]:token[1]]
        passages.append(passage)
        ids.append(answers[index])

query_embedding = model.encode(ids, device=device, show_progress_bar=True)
passage_embedding = model.encode(passages, device=device, show_progress_bar=True)

with open('query_embedding_1.pickle', 'wb') as pkl:
    pickle.dump({'ids': ids, 'embeddings': passage_embedding}, pkl, protocol=pickle.HIGHEST_PROTOCOL)






#print("questions: \t ,", questions)
#print("\n \n")
#print("answers: \t ,", answers)

# 按照行读取json, 得到一个dict, key只有一个 questions, 对应value: 是一个list
# 这个list 每一个元素 都是一个dict, 里面包含一个 'text', 'answer', 'page' 'tokenization' 等keys

# info['questions'][0]['text'] --> 这是第一个例子的 text
# info['questions'][0]['page']
# info['questions'][0]['tokenizations'] --> [[0, 132], [133, 206],
# [207, 286], [287, 425], [426, 516], [517, 589], [590, 723], [724, 800]]

"""
ids = []
vectors = []
with open(path) as f_in:
    for line in f_in:
        info = json.loads(line) # info: dictionary
        # print("info: \n", info)

        # print("info['questions']: \t", info['questions'])
        # print("info['questions'][0]: \t", info['questions'][0])  # 第一个 字典

        print("info['questions'][0]['tokenizations']: \t", info['questions'][0]['tokenizations'])

        docid = info['id']
        vector = info['contents']
        vectors.append(vector)
        ids.append(docid)
"""