from helper import *
from utils import *
from metrics import evaluate  # Evaluation metrics
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm

ave = True


# ave = False

def HAC_getClusters(params, embed1, embed2, cluster_threshold_real, mode, dim_is_bert=False):

    embed_dim = 300
    if mode == 'view-mean':
        dist1 = pdist(embed1, metric=params.metric)
        dist2 = pdist(embed2, metric=params.metric)
        dist = (dist1 + dist2) / 2
        if params.dataset == 'reverb45k':
            if not np.all(np.isfinite(dist)):
                for i in range(len(dist)):
                    if not np.isfinite(dist[i]):
                        dist[i] = 0
        clust_res = linkage(dist, method=params.linkage)
        if cluster_threshold_real == 'threshold':
            labels = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1
        else:
            labels = fcluster(clust_res, t=cluster_threshold_real, criterion='maxclust') - 1

        return labels, None
    else:
        dist = pdist(embed1, metric=params.metric)

        if params.dataset == 'reverb45k':
            if not np.all(np.isfinite(dist)):
                for i in range(len(dist)):
                    if not np.isfinite(dist[i]):
                        dist[i] = 0
        clust_res = linkage(dist, method=params.linkage)
        if cluster_threshold_real == 'threshold':
            labels = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1
        else:
            labels = fcluster(clust_res, t=cluster_threshold_real, criterion='maxclust') - 1
        clusters = [[] for i in range(max(labels) + 1)]
        for i in range(len(labels)):
            clusters[labels[i]].append(i)

        clusters_center = np.zeros((len(clusters), embed_dim), np.float32)
        for i in range(len(clusters)):
            cluster = clusters[i]
            if ave:
                clusters_center_embed = np.zeros(embed_dim, np.float32)
                for j in cluster:
                    embed_ = embed1[j]
                    clusters_center_embed += embed_
                clusters_center_embed_ = clusters_center_embed / len(cluster)
                clusters_center[i, :] = clusters_center_embed_
            else:
                sim_matrix = np.empty((len(cluster), len(cluster)), np.float32)
                for i in range(len(cluster)):
                    for j in range(len(cluster)):
                        if i == j:
                            sim_matrix[i, j] = 1
                        else:
                            if params.metric == 'cosine':
                                sim = cos_sim(embed1[i], embed1[j])
                            else:
                                sim = np.linalg.norm(embed1[i] - embed1[j])
                            sim_matrix[i, j] = sim
                            sim_matrix[j, i] = sim
                sim_sum = sim_matrix.sum(axis=1)
                max_num = cluster[int(np.argmax(sim_sum))]
                clusters_center[i, :] = embed1[max_num]
    # print('clusters_center:', type(clusters_center), clusters_center.shape)
        return labels, clusters_center


def cluster_test(params, side_info, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=False):
    sub_cluster_predict_list = []
    clust2ent = {}
    isSub = side_info.isSub
    triples = side_info.triples
    ent2id = side_info.ent2id

    for eid in isSub.keys():
        sub_cluster_predict_list.append(cluster_predict_list[eid])

    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}
    for rep, cluster in clust2ent.items():
        # cesi_clust2ent[rep] = list(cluster)
        cesi_clust2ent[rep] = set(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')

    cesi_ent2clust_u = {}
    if params.use_assume:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]
    else:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple_unique'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]

    cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')

    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, true_ent2clust, true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results[
        'pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results[
        'pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(true_clust2ent)
    gold_Singletons = len([1 for _, clust in true_clust2ent.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

    return ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, \
           macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons


def trans_embed(side_info, entity_embedding, relation_embedding, bert_embedding, clean_ent_list):
    relational_view_embed, semantic_view_embed = [], []
    for ent in clean_ent_list:
        id = side_info.ent2id[ent]
        if id in side_info.isSub:
            relational_view_embed.append(entity_embedding[id])
            semantic_view_embed.append(bert_embedding[id])
    print('relational_view_embed:', len(relational_view_embed))
    print('semantic_view_embed:', len(semantic_view_embed))

    return relational_view_embed, semantic_view_embed


def cluster_result(p, side_info, view_embed1, view_embed2, cluster_threshold_real, true_ent2clust, true_clust2ent,
                   mode):
    print(mode)

    print('cluster_threshold_real:', cluster_threshold_real)
    labels, clusters_center = HAC_getClusters(p, view_embed1, view_embed2, cluster_threshold_real, mode, True)

    cluster_predict_list = list(labels)
    ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, \
    pair_recall, macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons \
        = cluster_test(p, side_info, cluster_predict_list, true_ent2clust,
                       true_clust2ent)
    print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
          'pair_prec=', pair_prec)
    print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
          'pair_recall=', pair_recall)
    print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
    print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
    print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
    print()


def seed_accuracy(seed_pairs, side_info, true_ent2clust):
    total_num = len(seed_pairs)

    true_ent2clust_withoutwiki = dict()
    for k, v in true_ent2clust.items():
        k = k.split('|')[0]
        true_ent2clust_withoutwiki[k] = v

    num = 0
    for seed_pair in seed_pairs:
        sub0, sub1 = seed_pair
        sub0, sub1 = side_info.id2sub[sub0], side_info.id2sub[sub1]
        if sub0 in true_ent2clust_withoutwiki.keys() and sub1 in true_ent2clust_withoutwiki.keys():
            cluster0 = true_ent2clust_withoutwiki[sub0]
            cluster1 = true_ent2clust_withoutwiki[sub1]
            if cluster0 == cluster1:
                num += 1

    seed_accuracy = num / total_num
    return seed_accuracy


def score_result(seed_scores, side_info, true_ent2clust, threshold):
    total_num = len(seed_scores.keys())
    true_pos = 0
    fal_pos = 0
    fal_neg = 0
    true_neg = 0
    true_ent2clust_withoutwiki = dict()
    for k, v in true_ent2clust.items():
        k = k.split('|')[0]
        true_ent2clust_withoutwiki[k] = v

    for key, val in seed_scores.items():
        sub0, sub1 = side_info.id2sub[key[0]], side_info.id2sub[key[1]]
        if sub0 in true_ent2clust_withoutwiki.keys() and sub1 in true_ent2clust_withoutwiki.keys():
            cluster0 = true_ent2clust_withoutwiki[sub0]
            cluster1 = true_ent2clust_withoutwiki[sub1]
        if cluster0 == cluster1:
            if val >= threshold:
                true_pos += 1
            else:
                fal_neg += 1
        else:
            if val >= threshold:
                fal_pos += 1
            else:
                true_neg += 1

    precision = true_pos / (true_pos + fal_pos)
    recall = true_pos / (true_pos + fal_neg)
    accuracy = (true_pos + true_neg) / total_num

    print("threshold : ", threshold, "precision : ", precision, "recall : ", recall, "accuracy : ", accuracy)
