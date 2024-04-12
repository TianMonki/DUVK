import gensim, itertools, pickle, time
import torch

from helper import *
from model_TransE import KGEModel
from model_cross_encoder import Bert_Model_Cross
from utils import cos_sim
from test_performance import cluster_test, HAC_getClusters, trans_embed, cluster_result, seed_accuracy, score_result
from train_embedding_model import Train_Embedding_Model, pair2triples
from model_Bert import Bert_Model
import markov_clustering as mc


def get_seed_pair_list(embedding, side_info, k):
    # 2138*5
    sub_len = len(side_info.sub_list)
    topks, topk_cosines, cosines = getTopk(embedding, k)
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
            # pair = set()
            # # pair = (i, topks[i][j])
            # pair.add(i)
            # pair.add(topks[i][j])
            seed_pair_list.add(pair)
            seed_pair_list_uni.add(pair_uni)
            # seed_pair_list.add(pair)
            # seed_pair_cos[tuple(pair)] = cosines[i][j]
            # seed_pair_cos[pair_uni] = topk_cosines[i][j]
        for sub in range(i + 1, sub_len):
            seed_pair = (i, sub)
            seed_pair_cos[seed_pair] = cosines[i][sub]
    seed_pair_list = sorted(list(seed_pair_list))
    seed_pair_list_uni = sorted(list(seed_pair_list_uni))
    # seed_pair_list = sorted(seed_pair_list,key=lambda t:t[1])

    return seed_pair_list, seed_pair_list_uni, seed_pair_cos


def seed_pair2cluster(seed_pair_list, ent_list):
    pair_dict = dict()
    for seed_pair in seed_pair_list:
        rep, ent_id = seed_pair
        if ent_id not in pair_dict:
            if rep not in pair_dict:
                pair_dict.update({ent_id: rep})
            else:
                new_rep = pair_dict[rep]
                j = 0
                while rep in pair_dict:
                    new_rep = pair_dict[rep]
                    rep = new_rep
                    j += 1

                pair_dict.update({ent_id: new_rep})
        else:
            if rep not in pair_dict:
                new_rep = pair_dict[ent_id]
                if rep > new_rep:
                    pair_dict.update({rep: new_rep})
                else:
                    pair_dict.update({new_rep: rep})
            else:
                old_rep = rep
                new_rep = pair_dict[rep]
                j = 0
                while rep in pair_dict:
                    new_rep = pair_dict[rep]
                    rep = new_rep
                    j += 1

                if old_rep > new_rep:
                    pair_dict.update({ent_id: new_rep})
                else:
                    pair_dict.update({ent_id: old_rep})

    cluster_list = []
    for i in range(len(ent_list)):
        cluster_list.append(i)
    for ent_id in pair_dict:
        rep = pair_dict[ent_id]
        cluster_list[ent_id] = rep
    return cluster_list


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

        self.rel_id2sentence_list = dict()

        ent_id2sentence_list = self.side_info.ent_id2sentence_list
        for rel in self.side_info.rel_list:
            rel_id = self.side_info.rel2id[rel]
            if rel_id not in self.rel_id2sentence_list:
                triple_id_list = self.side_info.rel2triple_id_list[rel]
                sentence_list = []
                for triple_id in triple_id_list:
                    triple = self.triples_list[triple_id]
                    sub, rel_, obj = triple['triple'][0], triple['triple'][1], triple['triple'][2]
                    assert str(rel_) == str(rel)
                    if sub in self.side_info.ent2id:
                        sentence_list += ent_id2sentence_list[self.side_info.ent2id[sub]]
                    if obj in self.side_info.ent2id:
                        sentence_list += ent_id2sentence_list[self.side_info.ent2id[obj]]
                sentence_list = list(set(sentence_list))
                self.rel_id2sentence_list[rel_id] = sentence_list
        print('self.rel_id2sentence_list:', type(self.rel_id2sentence_list), len(self.rel_id2sentence_list))

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

        ''' Intialize embeddings '''
        if self.p.embed_init == 'crawl':
            fname1, fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/1E_init', '../file/' + self.p.dataset + '_' + self.p.split + '/1R_init'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate pre-trained embeddings')

                model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
                self.E_init = getEmbeddings(model, clean_ent_list, self.p.embed_dims)
                self.R_init = getEmbeddings(model, clean_rel_list, self.p.embed_dims)

                pickle.dump(self.E_init, open(fname1, 'wb'))
                pickle.dump(self.R_init, open(fname2, 'wb'))
            else:
                print('load init embeddings')
                self.E_init = pickle.load(open(fname1, 'rb'))
                self.R_init = pickle.load(open(fname2, 'rb'))

        else:
            print('generate init random embeddings')
            self.E_init = np.random.rand(len(clean_ent_list), self.p.embed_dims)
            self.R_init = np.random.rand(len(clean_rel_list), self.p.embed_dims)

        folder = 'multi_view/relation_view'
        print('folder:', folder)
        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)

        if self.p.use_Embedding_model:
            fname1, fname2 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/entity_embedding', '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/relation_embedding'
            if not checkFile(fname1) or not checkFile(fname2):
                print('generate TransE embeddings', fname1)
                entity_embedding, relation_embedding = self.E_init, self.R_init
                print('self.training_time', 'use pre-trained crawl embeddings ... ')

                TEM = Train_Embedding_Model(self.p, self.side_info, entity_embedding, relation_embedding,
                                            None, sub_index_list)
                self.entity_embedding, self.sub_embedding, self.relation_embedding = TEM.train()

                pickle.dump(self.entity_embedding, open(fname1, 'wb'))
                pickle.dump(self.relation_embedding, open(fname2, 'wb'))

            else:
                print('load TransE embeddings')
                self.entity_embedding = pickle.load(open(fname1, 'rb'))
                self.relation_embedding = pickle.load(open(fname2, 'rb'))

            TEM = Train_Embedding_Model(self.p, self.side_info, self.entity_embedding, self.relation_embedding,
                                        None, sub_index_list)
            # for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.entity_embedding[id]
            # for id in self.side_info.id2sub.keys(): self.sub2embed[id] = self.entity_embedding[id]
            # for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.relation_embedding[id]

        else:  # do not use embedding model
            for id in self.side_info.id2ent.keys(): self.ent2embed[id] = self.E_init[id]
            for id in self.side_info.id2sub.keys(): self.sub2embed[id] = self.E_init[id]
            for id in self.side_info.id2rel.keys(): self.rel2embed[id] = self.R_init[id]

        folder = 'multi_view/semantic_view_' + str(self.p.input)
        folder_to_make = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/'
        if not os.path.exists(folder_to_make):
            os.makedirs(folder_to_make)
        print('self.p.input:', self.p.input)

        print()
        self.epochs = 3
        # self.epochs = 1  # this is the rebuttal mode
        print('self.epochs:', self.epochs)

        if self.p.use_semantic and self.p.use_BERT:
            fname1 = '../file/' + self.p.dataset + '_' + self.p.split + '/' + folder + '/bert_embedding'
            if not checkFile(fname1):
                bert_model = Bert_Model(self.p, self.side_info)
                self.bert_embedding = list(bert_model.encode().values())
                pickle.dump(self.bert_embedding, open(fname1, 'wb'))
            else:
                print('load Bert embeddings')
                bert_model = Bert_Model(self.p, self.side_info)
                self.bert_embedding = pickle.load(open(fname1, 'rb'))

        nentity, nrelation = len(self.side_info.ent_list), len(self.side_info.rel_list)

        relational_view_embed, semantic_view_embed = trans_embed(self.side_info, self.entity_embedding,
                                                                 self.relation_embedding, self.bert_embedding,
                                                                 clean_ent_list)

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

        rel_seed, rel_seed_uni, rel_cosine_dict = get_seed_pair_list(semantic_view_embed,
                                                                     self.side_info,
                                                                     self.k)
        sem_seed, sem_seed_uni, sem_cosine_dict = get_seed_pair_list(relational_view_embed,
                                                                     self.side_info,
                                                                     self.k)
        rel_seed_acc = seed_accuracy(rel_seed_uni, self.side_info, self.true_ent2clust)
        print('rel_seed: ', len(rel_seed_uni))
        print('rel_seed_acc : ', rel_seed_acc)

        sem_seed_acc = seed_accuracy(sem_seed_uni, self.side_info, self.true_ent2clust)
        print('sem_seed: ', len(sem_seed_uni))
        print('sem_seed_acc : ', sem_seed_acc)

        self.intersection_seed = sorted(list(set(rel_seed_uni).intersection(set(sem_seed_uni))))
        print('intersection_seed: ', len(self.intersection_seed))
        intersection_seed_acc = seed_accuracy(self.intersection_seed, self.side_info, self.true_ent2clust)
        print('Intersection_seed_acc : ', intersection_seed_acc)

        self.union_seed = sorted(list(set(rel_seed_uni).union(set(sem_seed_uni))))
        print('union_seed: ', len(self.union_seed))
        union_seed_acc = seed_accuracy(self.union_seed, self.side_info, self.true_ent2clust)
        print('Union_seed_acc : ', union_seed_acc)

        # # cluster_result(self.p, self.side_info, relational_view_embed, None, self.c_num, self.true_ent2clust,
        # #                self.true_clust2ent, mode='relational-view')
        # # cluster_result(self.p, self.side_info, semantic_view_embed, None, self.c_num, self.true_ent2clust,
        # #                self.true_clust2ent, mode='semantic-view')

        if not checkFile(fname1) or not checkFile(fname2) or not checkFile(fname3):
            for epoch in range(self.epochs):
                print('exchange seed epoch : ', epoch)
                print('relational view seed :')
                seed_trpIds, seed_sim = pair2triples(rel_seed_uni, clean_ent_list, self.side_info.ent2id,
                                                     self.side_info.id2ent, self.side_info.ent2triple_id_list,
                                                     self.side_info.trpIds, self.entity_embedding, cos_sim,
                                                     is_cuda=False,
                                                     high_confidence=False)

                print('relational view seed-train: ')
                rel_topks, rel_topk_cosines, rel_cosines = getTopk(semantic_view_embed, self.k)
                sem_topks, sem_topk_cosines, sem_cosines = getTopk(relational_view_embed, self.k)

                print('seed_trpIds:', type(seed_trpIds), len(seed_trpIds), seed_trpIds[0:30])
                print('seed_sim:', type(seed_sim), len(seed_sim), seed_sim[0:30])
                print()

                kge_model = KGEModel(
                    model_name=self.p.model,
                    dict_local=self.p.embed_loc,
                    init=self.p.embed_init,
                    E_init=self.entity_embedding,
                    R_init=self.relation_embedding,
                    nentity=nentity,
                    nrelation=nrelation,
                    hidden_dim=self.p.hidden_dim,
                    gamma=self.p.single_gamma,
                    double_entity_embedding=self.p.double_entity_embedding,
                    double_relation_embedding=self.p.double_relation_embedding
                )

                self.entity_embedding, self.sub_embedding, self.relation_embedding = TEM.cross_train(kge_model,
                                                                                                     seed_trpIds,
                                                                                                     seed_sim,
                                                                                                     rel_topks)
                pickle.dump(self.entity_embedding, open(fname1, 'wb'))
                pickle.dump(self.relation_embedding, open(fname2, 'wb'))
                print('semantic view seed-train: ')
                sem_input_list = clean_sub_list
                self.bert_embedding = bert_model.fine_tune(sem_input_list, sem_seed, sem_topks)
                pickle.dump(self.bert_embedding, open(fname3, 'wb'))

                relational_view_embed, semantic_view_embed = trans_embed(self.side_info, self.entity_embedding,
                                                                         self.relation_embedding,
                                                                         self.bert_embedding, clean_ent_list)

                cluster_result(self.p, self.side_info, relational_view_embed, None, self.c_num, self.true_ent2clust,
                               self.true_clust2ent, mode='relational-view')
                cluster_result(self.p, self.side_info, semantic_view_embed, None, self.c_num, self.true_ent2clust,
                               self.true_clust2ent, mode='semantic-view')

                rel_seed, rel_seed_uni, rel_cosine_dict = get_seed_pair_list(semantic_view_embed,
                                                                             self.side_info,
                                                                             self.k)

                sem_seed, sem_seed_uni, sem_cosine_dict = get_seed_pair_list(relational_view_embed,
                                                                             self.side_info,
                                                                             self.k)

                rel_seed_acc = seed_accuracy(rel_seed_uni, self.side_info, self.true_ent2clust)
                print('rel_seed_len : ', len(rel_seed_uni))
                print('rel_seed_acc : ', rel_seed_acc)

                sem_seed_acc = seed_accuracy(sem_seed_uni, self.side_info, self.true_ent2clust)
                print('sem_seed_len : ', len(sem_seed_uni))
                print('sem_seed_acc : ', sem_seed_acc)

                self.union_seed = sorted(list(set(rel_seed_uni).union(set(sem_seed_uni))))
                print('union_seed_len : ', len(self.union_seed))
                union_seed_acc = seed_accuracy(self.union_seed, self.side_info, self.true_ent2clust)
                print('union_seed_acc : ', union_seed_acc)
                self.intersection_seed = sorted(list(set(rel_seed_uni).intersection(set(sem_seed_uni))))

                print('intersection_seed_len : ', len(self.intersection_seed))
                intersection_seed_acc = seed_accuracy(self.intersection_seed, self.side_info, self.true_ent2clust)
                print('intersection_seed_acc : ', intersection_seed_acc)

        else:
            self.entity_embedding = pickle.load(open(fname1, 'rb'))
            self.relation_embedding = pickle.load(open(fname2, 'rb'))
            self.bert_embedding = pickle.load(open(fname3, 'rb'))
            relational_view_embed, semantic_view_embed = trans_embed(self.side_info, self.entity_embedding,
                                                                     self.relation_embedding,
                                                                     self.bert_embedding,
                                                                     clean_ent_list)

            # rel_topks, rel_topk_cosines, rel_cosines = getTopk(semantic_view_embed, self.k)
            # sem_topks, sem_topk_cosines, sem_cosines = getTopk(relational_view_embed, self.k)
            rel_seed, rel_seed_uni, rel_cosine_dict = get_seed_pair_list(semantic_view_embed,
                                                                         self.side_info,
                                                                         self.k)
            sem_seed, sem_seed_uni, sem_cosine_dict = get_seed_pair_list(relational_view_embed,
                                                                         self.side_info,
                                                                         self.k)

            # rel_seed_acc = seed_accuracy(rel_seed_cmvc, self.side_info, self.true_ent2clust)
            # print('rel_seed: ', len(rel_seed_cmvc))
            # print('rel_seed_acc : ', rel_seed_acc)
            # sem_seed_acc = seed_accuracy(sem_seed_cmvc, self.side_info, self.true_ent2clust)
            # print('sem_seed: ', len(sem_seed_cmvc))
            # print('sem_seed_acc : ', sem_seed_acc)

            self.union_seed = sorted(list(set(rel_seed_uni).union(set(sem_seed_uni))))
            # union_seed_acc = seed_accuracy(self.union_seed, self.side_info, self.true_ent2clust)
            # print('union_seed: ', len(self.union_seed))
            # print('Union_seed_acc : ', union_seed_acc)
            self.intersection_seed = sorted(list(set(rel_seed_uni).intersection(set(sem_seed_uni))))
            # intersection_seed_cmvc = sorted(list(set(self.intersection_seed).union(set(cmvc_seed_list_con))))

            # print('intersection_seed: ', len(self.intersection_seed))
            # intersection_seed_acc = seed_accuracy(self.intersection_seed, self.side_info, self.true_ent2clust)
            # print('Intersection_seed_acc : ', intersection_seed_acc)

        #   cross-encoder
        self.all_pair_cosines, self.intersection_cosines, self.union_cosines = merge_dicts(rel_cosine_dict,
                                                                                           sem_cosine_dict,
                                                                                           self.intersection_seed,
                                                                                           self.union_seed)

        fname1 = '../output/state_dict/multi_view/' + self.p.dataset + '/union_seed_score'
        fname2 = '../output/state_dict/multi_view/' + self.p.dataset + '/crossencoder.pth'
        if not checkFile(fname1):
            bert_model_cross = Bert_Model_Cross(self.p, self.side_info, self.intersection_seed,
                                                self.intersection_cosines, self.union_seed)
            bert_model_cross.fine_tune()
            # bert_model_cross.load_state()
            print('cross_encoder train finished')
            pair_score = bert_model_cross.inference()
            pickle.dump(pair_score, open(fname1, 'wb'))
        else:
            pair_score = pickle.load(open(fname1, 'rb'))

        adjacency_matrix = np.zeros((len(self.side_info.sub_list), len(self.side_info.sub_list)))
        for key, val in pair_score.items():
            adjacency_matrix[key[0], key[1]] = val
            adjacency_matrix[key[1], key[0]] = val

        # normalized_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0)

        cluster_res = mc.run_mcl(adjacency_matrix, expansion=3, inflation=3)
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
