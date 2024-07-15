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
    for phr in phr_list:
        if phr in model.vocab:
            embed_list.append(model.word_vec(phr))
            all_num += 1
        else:
            vec = np.zeros(embed_dims, np.float32)
            wrds = word_tokenize(phr)
            for wrd in wrds:
                all_num += 1
                if wrd in model.vocab:
                    vec += model.word_vec(wrd)
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
    max_topks, cosines = [], []
    embedding = np.array(embedding)
    for i in range(len(embedding)):
        target_embedding = embedding[i].reshape(1, -1)
        cos_sim = cosine_similarity(embedding, target_embedding)
        cos_sim = cos_sim.tolist()
        cosines.append(cos_sim)

        max_topk = list(map(cos_sim.index, heapq.nlargest(k + 1, cos_sim)))
        max_topk.remove(i)
        max_topks.append(max_topk)

    return max_topks, cosines

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
