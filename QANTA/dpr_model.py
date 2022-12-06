from qanta_util.qbdata import QantaDatabase
from sentence_transformers import SentenceTransformer, util
import pickle
#import torch
import json

passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')

path = "./qantatest.json"
guesstrain = QantaDatabase(path)

questions = [x.text for x in guesstrain.guess_test_questions]
answers = [x.page for x in guesstrain.guess_test_questions]
tokens = [x.tokenizations for x in guesstrain.guess_test_questions]

print(len(questions))
#print(answers)
print(len(tokens))

ids = []
passages = []

for index in range(len(questions)):
    question = questions[index]
    for token in tokens[index]:
        passage = question[token[0]:token[1]]
        passages.append(passage)
        ids.append(answers[index])


passage_embedding = passage_encoder.encode(passages)

with open('passage_embedding_2.pickle', 'wb') as pkl:
    pickle.dump({'ids': ids, 'embeddings': passage_embedding}, pkl, protocol=pickle.HIGHEST_PROTOCOL)