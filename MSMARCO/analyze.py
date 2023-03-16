import pickle
import csv
import json
from sentence_transformers import SentenceTransformer, util
import torch

def check_mrr(dict1):
    answer_dict = dict()
    with open("./data/qrels.dev.small.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in answer_dict:
                answer_dict[line[0]] = []
            answer_dict[line[0]].append(line[2])
    dict2 = answer_dict
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

def check_cell(dict1, dict5):
    answer_dict = dict()
    with open("./data/qrels.dev.small.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in answer_dict:
                answer_dict[line[0]] = []
            answer_dict[line[0]].append(line[2])
    with open('./result/cell_index_list.pickle', 'rb') as fp:
        cache_data = pickle.load(fp)
    centroid_dict = dict()
    for i in cache_data:
        centroid_dict[i[1]] = i[0][0][0]
    dict2 = answer_dict
    dict4 = centroid_dict
    total_pair=0
    score = 0
    for i in dict1.keys():
        if i in dict2:
            total_pair += 1

            list2 = dict2[i]
            curr_cen = int(dict5[i][1:-1])
            same_cen = False
            for answer in list2:
                centroid = int(dict4[int(answer)])
                if centroid == curr_cen:
                    same_cen = True
            if same_cen:
                score += 1
    return score/total_pair

def check_rank(dict1):
    predict_dict = dict()
    with open("./result/exact_match.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in predict_dict:
                   predict_dict[line[0]] = []
            predict_dict[line[0]].append(line[1])
    answer_dict = dict()
    with open("./data/qrels.dev.small.tsv") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in answer_dict:
                answer_dict[line[0]] = []
            answer_dict[line[0]].append(line[2])
    dict3 = predict_dict
    dict2 = answer_dict
    total_pair1 = 0
    counter1 = 0
    for i in dict1.keys():
        if i in dict2:
            list1 = dict1[i]
            list2 = dict2[i]
            list3 = dict3[i]
            index1 = len(list1)
            index2 = len(list3)
            found = False
            for answer in list2:
                if answer in list3 and answer in list1:
                    found = True
                if answer in list1:
                    index1 = min(index1, list1.index(answer))
                if answer in list3:
                    index2 = min(index2, list3.index(answer))
            if found:
                total_pair1 += 1
                if index1>index2:
                    counter1 += 1
    return counter1/total_pair1


file_name = ["ivf102410.tsv"]
result_list = []
#centroid_predict_dict1 = dict()
for i in file_name:
    centroid_predict_dict = dict()
    with open(i) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[0] not in centroid_predict_dict:
                centroid_predict_dict[line[0]] = []
            centroid_predict_dict[line[0]].append(line[1])
            #centroid_predict_dict1[line[0]] = line[3]

print("MRR score: ", check_mrr(centroid_predict_dict))
#print("How many correct answers have different cell: ", check_cell(centroid_predict_dict, centroid_predict_dict1))
#print("How many FAISS result have lower rank:", check_rank(centroid_predict_dict))
