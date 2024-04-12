import os, sys
import numpy as np, json
from nltk.tokenize import word_tokenize
import pathlib
import heapq
from sklearn.metrics.pairwise import cosine_similarity


def checkFile(filename):
    return pathlib.Path(filename).is_file()


def invertDic(my_map, struct='o2o'):
    inv_map = {}

    if struct == 'o2o':  # Reversing one-to-one dictionary
        for k, v in my_map.items():
            inv_map[v] = k

    elif struct == 'm2o':  # Reversing many-to-one dictionary
        for k, v in my_map.items():
            inv_map[v] = inv_map.get(v, [])
            inv_map[v].append(k)

    elif struct == 'm2ol':  # Reversing many-to-one list dictionary
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, [])
                inv_map[ele].append(k)

    elif struct == 'm2os':
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, set())
                inv_map[ele].add(k)

    elif struct == 'ml2o':  # Reversing many_list-to-one dictionary
        for k, v in my_map.items():
            for ele in v:
                inv_map[ele] = inv_map.get(ele, [])
                inv_map[ele] = k
    return inv_map


# Get embedding of words from gensim word2vec model
# clean_list == phr_list
def getEmbeddings(model, phr_list, embed_dims):
    embed_list = []
    all_num, oov_num, oov_rate = 0, 0, 0
    for i in range(len(phr_list)):
        if phr_list[i] in model.index2entity:
            embed_list.append(model.vectors[i])
            all_num += 1
        else:
            vec = np.zeros(embed_dims, np.float32)
            wrds = word_tokenize(phr_list[i])
            for w in range(len(wrds)):
                all_num += 1
                if wrds[w] in model.index2entity:
                    vec += model.vectors[w]
                else:
                    vec += np.random.randn(embed_dims)
                    oov_num += 1
            if len(wrds) == 0:
                embed_list.append(vec / 10000)
            else:
                embed_list.append(vec / len(wrds))
    oov_rate = oov_num / all_num
    print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
    return np.array(embed_list)


# embedding ndarray_list
def getTopk(embedding, k):
    max_topks, topk_cosines, cosines = [], [], []
    for i in range(len(embedding)):
        # embedding_cy = embedding.copy()
        # embedding_cy.pop(i)
        # print(np.array(embedding_cy).shape)
        cos_sim = cosine_similarity(np.array(embedding), embedding[i].reshape(1, -1)).tolist()
        cosines.append(cos_sim)

        cosine = heapq.nlargest(k + 1, cos_sim)
        cosine.pop(0)
        cosine = [i for j in cosine for i in j]
        topk_cosines.append(cosine)
        max_topk = list(map(cos_sim.index, heapq.nlargest(k + 1, cos_sim)))
        max_topk.remove(i)
        max_topks.append(max_topk)
    return max_topks, topk_cosines, cosines

def union_topk(str_topks, con_topks):
    union_topks = []
    for s, c in str_topks, con_topks:
        union_topks.append(list(set(s).union(set(c))))
    return union_topks

def intersection_topk(str_topks, con_topks):
    intersection_topks = []
    for i in range(len(str_topks)):
        intersection_topks.append(list(set(str_topks[i]).intersection(set(con_topks[i]))))

    return intersection_topks

def merge_dicts(dict1, dict2, intersection_seed, union_seed):
    all_result = dict()
    intersection_result = dict()
    union_result = dict()
    for k, v in dict1.items():
        all_result[k] = (v[0] + dict2[k][0]) / 2

    for i in intersection_seed:
        intersection_result[i] = all_result[i]
    for j in union_seed:
        union_result[j] = all_result[j]

    return all_result, intersection_result, union_result
