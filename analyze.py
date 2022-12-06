import pickle
import csv
import json
from sentence_transformers import SentenceTransformer, util
import torch
from matplotlib import pyplot as plt

def check_mrr(dict1, dict2):
    total_score = 0
    total_pair = 0
    for i in dict1.keys():
        if i in dict2:
            total_pair += 1
            list1 = dict1[i]
            list2 = dict2[i]
            index = len(list1)
            score = 0
            for answer in list2:
                if answer in list1:
                    index = min(index, list1.index(answer))
            if index != len(list1):
                score = 1/(index+1)
            total_score += score
    return total_score/total_pair

def analyze_mrr(dict1, dict3, dict2, dict4, dict5):
    total_pair=0
    total_pair1 = 0
    counter1 = 0
    counter2 = 0
    chanegd_rank = 0
    score = 0
    for i in dict1.keys():
        if i in dict2:
            total_pair += 1

            list1 = dict1[i]
            list2 = dict2[i]
            list3 = dict3[i]
            curr_cen = int(dict5[i][1:-1])
            index1 = len(list1)
            index2 = len(list3)
            found = False
            same_cen = False
            for answer in list2:
                centroid = int(dict4[int(answer)])
                if centroid == curr_cen:
                    same_cen = True
                if answer in list3 and answer not in list1:
                    found = True
                if answer in list1:
                    index1 = min(index1, list1.index(answer))
                if answer in list3:
                    index2 = min(index2, list3.index(answer))
            if found:
                total_pair1 += 1
                if same_cen:
                    score += 1
                if index1>index2:
                    counter1 += 1
                if index2>index1:
                    counter2 += 1
    return score/total_pair1

#with open('cell_index_list.pickle', 'rb') as fp:
#    cache_data = pickle.load(fp)
#centroid_dict = dict()
#for i in cache_data:
#    centroid_dict[i[1]] = i[0][0][0]
answer_dict = dict()
with open("data/qrels.dev.small.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        if line[0] not in answer_dict:
            answer_dict[line[0]] = []
        answer_dict[line[0]].append(line[2])

#centroid_predict_dict1 = dict()
file_name = ["output.dev.tsv", "2probe.tsv", "3probe.tsv", "4probe.tsv", "5probe.tsv", "10probe.tsv", "exact_match.tsv"]
result_list = []
for i in file_name:
    centroid_predict_dict = dict()
    with open(i) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in centroid_predict_dict:
                centroid_predict_dict[line[0]] = []
            centroid_predict_dict[line[0]].append(line[1])
    score = check_mrr(centroid_predict_dict, answer_dict)
    result_list.append(score)
print(result_list)
plt.plot(["1","2","3","4","5","10","exact"],result_list, linestyle='--', marker='o', color='b', label='line with marker')
plt.xlabel('Number of probe')
plt.ylabel('MRR@10')
plt.title("MRR@10 Score for different probe")
plt.savefig("result.png")
#        centroid_predict_dict1[line[0]] = line[3]
#predict_dict = dict()
#with open("exact_match.tsv") as file:
#    tsv_file = csv.reader(file, delimiter="\t")
#    for line in tsv_file:
#        if line[0] not in predict_dict:
#               predict_dict[line[0]] = []
#        predict_dict[line[0]].append(line[1])


#print(analyze_mrr(centroid_predict_dict, predict_dict, answer_dict,centroid_dict, centroid_predict_dict1 ))
