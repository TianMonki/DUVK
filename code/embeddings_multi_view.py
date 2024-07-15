import gensim, itertools, pickle, time
import torch

from helper import *
from test_performance import cluster_test, HAC_getClusters, trans_embed, cluster_result, seed_accuracy, score_result
import markov_clustering as mc


def get_seed_pair_list(embedding, side_info, k):
    sub_len = len(side_info.sub_list)
    topks, cosines = getTopk(embedding, k)
    seed_pair_cos = dict()
    seed_pair_list_uni = set()
    seed_pair_list = set()
    for i in range(sub_len):
        for j in range(k):
            pair = (i, topks[i][j])
            if i < topks[i][j]:
                pair_uni = (i, topks[i][j])
            else:
                pair_uni = (topks[i][j], i)
            seed_pair_list.add(pair)
            seed_pair_list_uni.add(pair_uni)

        for sub in range(i + 1, sub_len):
            seed_pair = (i, sub)
            seed_pair_cos[seed_pair] = cosines[i][sub]

    seed_pair_list = sorted(list(seed_pair_list))
    seed_pair_list_uni = sorted(list(seed_pair_list_uni))

    return seed_pair_list, seed_pair_list_uni, seed_pair_cos, topks, cosines


class Embeddings(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, true_ent2clust, true_clust2ent, sub_uni2triple_dict=None,
                 triple_list=None):
        self.p = params

        self.side_info = side_info
        self.ent2embed = {}  # Stores final embeddings learned for noun phrases
        self.rel2embed = {}  # Stores final embeddings learned for relation phrases
        self.sub2embed = {}
        self.true_ent2clust, self.true_clust2ent = true_ent2clust, true_clust2ent
        self.sub_uni2triple_dict = sub_uni2triple_dict
        self.triples_list = triple_list

    def fit(self):

        show_memory = False
        if show_memory:
            print('show_memory:', show_memory)
            import tracemalloc
            tracemalloc.start(25)  # 默认25个片段，这个本质还是多次采样

        clean_ent_list, clean_rel_list, clean_sub_list, sub_index_list = [], [], [], []
        for index in range(len(self.side_info.ent_list)):
            clean_ent = self.side_info.ent_list[index].split('|')[0]
            if clean_ent in self.side_info.sub_list:
                sub_index_list.append(index)
                clean_sub_list.append(clean_ent)
            clean_ent_list.append(clean_ent)

        for rel in self.side_info.rel_list: clean_rel_list.append(rel.split('|')[0])

        print('clean_ent_list:', type(clean_ent_list), len(clean_ent_list))
        print('clean_rel_list:', type(clean_rel_list), len(clean_rel_list))

        print()

        self.epochs = 3
        # self.epochs = 1  # this is the rebuttal mode
        print('self.epochs:', self.epochs)

        folder_to_make = '../output/state_dict/multi_view/' + self.p.dataset
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)
        fname1 = folder_to_make + '/entity_embedding'
        fname2 = folder_to_make + '/relation_embedding'
        fname3 = folder_to_make + '/bert_embedding'

        if self.p.dataset == 'OPIEC59k':
            self.k = 3
            self.c_num = 490
        else:
            self.k = 1
            self.c_num = 6700

        self.entity_embedding = pickle.load(open(fname1, 'rb'))
        self.relation_embedding = pickle.load(open(fname2, 'rb'))
        self.bert_embedding = pickle.load(open(fname3, 'rb'))
        self.relational_view_embed, self.semantic_view_embed = trans_embed(self.side_info, self.entity_embedding,
                                                                           self.relation_embedding,
                                                                           self.bert_embedding,
                                                                           clean_ent_list)

        self.rel_seed, self.rel_seed_uni, self.rel_cosine_dict, self.rel_topks, self.rel_cosines = get_seed_pair_list(
            self.semantic_view_embed,
            self.side_info,
            self.k)
        self.sem_seed, self.sem_seed_uni, self.sem_cosine_dict, self.sem_topks, self.sem_cosines = get_seed_pair_list(
            self.relational_view_embed,
            self.side_info,
            self.k)

        self.union_seed = sorted(list(set(self.rel_seed_uni).union(set(self.sem_seed_uni))))
        self.intersection_seed = sorted(list(set(self.rel_seed_uni).intersection(set(self.sem_seed_uni))))

        #   cross-encoder
        self.all_pair_cosines, self.intersection_cosines, self.union_cosines = merge_dicts(self.rel_cosine_dict,
                                                                                           self.sem_cosine_dict,
                                                                                           self.intersection_seed,
                                                                                           self.union_seed)

        fname1 = '../output/state_dict/multi_view/' + self.p.dataset + '/union_seed_score'
        fname2 = '../output/state_dict/multi_view/' + self.p.dataset + '/crossencoder.pth'

        pair_score = pickle.load(open(fname1, 'rb'))
        adjacency_matrix = np.zeros((len(self.side_info.sub_list), len(self.side_info.sub_list)))
        for key, val in pair_score.items():
            adjacency_matrix[key[0], key[1]] = val
            adjacency_matrix[key[1], key[0]] = val

        cluster_res = mc.run_mcl(adjacency_matrix, expansion=2, inflation=3)
        cluster_list = mc.get_clusters(cluster_res)

        clusters = []
        for i in range(len(self.side_info.sub_list)):
            clusters.append(i)
        for i in cluster_list:
            for j in i:
                clusters[j] = i[0]
        cluster_test(self.p, self.side_info, clusters, self.true_ent2clust,
                     self.true_clust2ent,
                     print_or_not=True)
        print('finished')
