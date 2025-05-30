import logging

from torch.utils.data import DataLoader
from helper import *
from utils import cos_sim, EarlyStopping
from dataloader_TransE import *
from model_TransE import KGEModel
import pickle


def pair2triples(seed_pair_list, ent_list, ent2id, id2ent, ent2triple_id_list, trpIds, entity_embedding, cos_sim,
                 is_cuda=False, high_confidence=False):
    seed_trpIds, seed_sim = [], []
    if is_cuda:
        entity_embed = entity_embedding.data
    else:
        entity_embed = entity_embedding

    for seed_pair in seed_pair_list:
        i, j = seed_pair[0], seed_pair[1]
        if i < len(ent_list) and j < len(ent_list):
            ent1, ent2 = ent_list[i], ent_list[j]
            e1_embed, e2_embed = entity_embed[i], entity_embed[j]
            if is_cuda:
                sim = torch.cosine_similarity(e1_embed, e2_embed, dim=0)
            else:
                if not np.dot(e1_embed, e2_embed) == 0:
                    sim = cos_sim(e1_embed, e2_embed)
                else:
                    sim = 0
            if high_confidence:
                if sim > 0.9:
                    Append = True
                else:
                    Append = False
            else:
                Append = True
            if Append:
                for ent in [ent1, ent2]:
                    triple_list = ent2triple_id_list[ent]
                    for triple_id in triple_list:
                        triple = trpIds[triple_id]
                        if str(id2ent[triple[0]]) == str(ent1):
                            trp = (ent2id[str(ent2)], triple[1], triple[2])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
                        if str(id2ent[triple[0]]) == str(ent2):
                            trp = (ent2id[str(ent1)], triple[1], triple[2])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
                        if str(id2ent[triple[2]]) == str(ent1):
                            trp = (triple[0], triple[1], ent2id[str(ent2)])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
                        if str(id2ent[triple[2]]) == str(ent2):
                            trp = (triple[0], triple[1], ent2id[str(ent1)])
                            seed_trpIds.append(trp)
                            seed_sim.append(sim)
    return seed_trpIds, seed_sim


class Train_Embedding_Model(object):
    """
    Learns embeddings for NPs and relation phrases
    """

    def __init__(self, params, side_info, E_init, R_init, seed_pair, sub_index_list):
        self.p = params
        self.side_info = side_info
        self.E_init = E_init
        self.R_init = R_init
        self.web_seed_pair_list = seed_pair
        self.sub_index_list = sub_index_list
        # self.new_seed_trpIds = new_seed_triples
        # self.new_seed_sim = new_seed_sim

    def __del__(self):
        print("Train_Embedding_Model del ... ")

    def train(self):
        KGEModel.set_logger(self)
        # num
        nentity, nrelation = len(self.side_info.ent_list), len(self.side_info.rel_list)
        train_triples = self.side_info.trpIds
        logging.info('#train: %d' % len(train_triples))

        self.nentity = nentity
        self.nrelation = nrelation

        logging.info('Model: %s' % self.p.model)
        logging.info('#entity: %d' % nentity)
        logging.info('#relation: %d' % nrelation)
        logging.info('#train: %d' % len(train_triples))

        kge_model = KGEModel(
            model_name=self.p.model,
            dict_local=self.p.embed_loc,
            init=self.p.embed_init,
            E_init=self.E_init,
            R_init=self.R_init,
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=self.p.hidden_dim,
            gamma=self.p.single_gamma,
            double_entity_embedding=self.p.double_entity_embedding,
            double_relation_embedding=self.p.double_relation_embedding
        )

        logging.info('Model Parameter Configuration:')
        for name, param in kge_model.named_parameters():
            logging.info(
                'Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

        if self.p.cuda:
            kge_model = kge_model.cuda()

        if self.p.do_train:
            # Set training dataloader iterator
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, self.p.single_negative_sample_size, 'head-batch'),
                batch_size=self.p.single_batch_size,
                shuffle=True,
                num_workers=max(1, self.p.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn
            )

            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, self.p.single_negative_sample_size, 'tail-batch'),
                batch_size=self.p.single_batch_size,
                shuffle=True,
                num_workers=max(1, self.p.cpu_num // 2),
                collate_fn=TrainDataset.collate_fn
            )

            self.train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

            # Set training configuration
            current_learning_rate = self.p.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            if self.p.warm_up_steps:
                warm_up_steps = self.p.warm_up_steps
            else:
                warm_up_steps = self.p.max_steps // 2

        if self.p.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % self.p.init_checkpoint)
            checkpoint = torch.load(os.path.join(self.p.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            kge_model.load_state_dict(checkpoint['model_state_dict'])
            if self.p.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # logging.info('Ramdomly Initializing %s Model...' % self.p.model)
            init_step = 0

        step = init_step

        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('single_batch_size = %d' % self.p.single_batch_size)
        logging.info('single_negative_adversarial_sampling = %d' % self.p.single_negative_sample_size)
        logging.info('hidden_dim = %d' % self.p.hidden_dim)
        logging.info('single_gamma = %f' % self.p.single_gamma)
        logging.info('negative_adversarial_sampling = %s' % str(self.p.negative_adversarial_sampling))
        if self.p.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % self.p.adversarial_temperature)
        if self.p.use_cross_seed:
            logging.info('self.p.use_cross_seed = %f' % self.p.use_cross_seed)
            logging.info('self.p.update_seed = %f' % self.p.update_seed)

            logging.info('self.p.max_steps = %f' % self.p.max_steps)
            logging.info('self.p.turn_to_seed = %f' % self.p.turn_to_seed)
            logging.info('self.p.seed_max_steps = %f' % self.p.seed_max_steps)
            logging.info('self.p.update_seed_steps = %f' % self.p.update_seed_steps)
        else:
            logging.info('Do not use seeds ...')

        # Set valid dataloader as it would be evaluated during training
        if self.p.do_train:
            logging.info('learning_rate = %d' % current_learning_rate)
            training_logs = []

        loss_list = []
        step_list = []

        for step in range(init_step, self.p.max_steps):
            log = kge_model.train_step(self.p, kge_model, optimizer, self.train_iterator)
            loss = log['loss']
            loss_list.append(loss)
            step_list.append(step)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % self.p.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                KGEModel.log_metrics(self.p, 'Training average', step, metrics)
                training_logs = []

        self.entity_embedding = kge_model.entity_embedding.detach().cpu().numpy()
        self.sub_embedding = self.entity_embedding[self.sub_index_list]
        self.relation_embedding = kge_model.relation_embedding.detach().cpu().numpy()

        return self.entity_embedding, self.sub_embedding, self.relation_embedding

    # exchange triples
    def cross_train(self, kge_model, new_seed_trpIds, new_seed_sim, new_topks):
        # num
        nentity, nrelation = len(self.side_info.ent_list), len(self.side_info.rel_list)

        self.nentity = nentity
        self.nrelation = nrelation
        self.new_seed_trpIds = new_seed_trpIds
        self.new_seed_sim = new_seed_sim
        self.new_topks = new_topks
        self.kge_model = kge_model

        KGEModel.set_logger(self)
        logging.info('Model Parameter Configuration:')
        for name, param in kge_model.named_parameters():
            logging.info(
                'Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

        if self.p.cuda:
            kge_model = kge_model.cuda()

        if self.p.do_train:
            # Set training configuration
            current_learning_rate = self.p.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
            if self.p.warm_up_steps:
                warm_up_steps = self.p.warm_up_steps
            else:
                warm_up_steps = self.p.max_steps // 2

        if self.p.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % self.p.init_checkpoint)
            checkpoint = torch.load(os.path.join(self.p.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            kge_model.load_state_dict(checkpoint['model_state_dict'])
            if self.p.do_train:
                current_learning_rate = checkpoint['current_learning_rate']
                warm_up_steps = checkpoint['warm_up_steps']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # logging.info('Ramdomly Initializing %s Model...' % self.p.model)
            init_step = 0

        step = init_step

        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('single_batch_size = %d' % self.p.single_batch_size)
        logging.info('single_negative_adversarial_sampling = %d' % self.p.single_negative_sample_size)
        logging.info('hidden_dim = %d' % self.p.hidden_dim)
        logging.info('single_gamma = %f' % self.p.single_gamma)
        logging.info('negative_adversarial_sampling = %s' % str(self.p.negative_adversarial_sampling))
        if self.p.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % self.p.adversarial_temperature)
        if self.p.use_cross_seed:
            logging.info('self.p.use_cross_seed = %f' % self.p.use_cross_seed)
            logging.info('self.p.update_seed = %f' % self.p.update_seed)

            logging.info('self.p.max_steps = %f' % self.p.max_steps)
            logging.info('self.p.turn_to_seed = %f' % self.p.turn_to_seed)
            logging.info('self.p.seed_max_steps = %f' % self.p.seed_max_steps)
            logging.info('self.p.update_seed_steps = %f' % self.p.update_seed_steps)
        else:
            logging.info('Do not use seeds ...')

        # Set valid dataloader as it would be evaluated during training

        if self.p.do_train:
            logging.info('learning_rate = %d' % current_learning_rate)

            training_logs = []
            # if self.p.use_cross_seed:
            if len(new_seed_trpIds) > 0:
                logging.info('#Seed: %d' % len(new_seed_trpIds))
                seed_trpIds = new_seed_trpIds
                seed_sim = new_seed_sim

            seed_dataloader_head = DataLoader(
                SeedDataset(seed_trpIds, self.side_info, nentity, nrelation, self.p.cross_negative_sample_size,
                            'head-batch', seed_sim, new_topks),
                batch_size=self.p.cross_batch_size,
                shuffle=True,
                num_workers=max(1, self.p.cpu_num // 2),
                collate_fn=SeedDataset.collate_fn
            )

            seed_dataloader_tail = DataLoader(
                SeedDataset(seed_trpIds, self.side_info, nentity, nrelation, self.p.cross_negative_sample_size,
                            'tail-batch', seed_sim, new_topks),
                batch_size=self.p.cross_batch_size,
                shuffle=True,
                num_workers=max(1, self.p.cpu_num // 2),
                collate_fn=SeedDataset.collate_fn
            )
            self.seed_iterator = BidirectionalOneShotIterator(seed_dataloader_head, seed_dataloader_tail)

            early_stopping = EarlyStopping(patience=3, delta=4)

            for step in range(init_step, 20000):
                if self.p.use_cross_seed:
                    log = kge_model.cross_train_step(self.p, kge_model, optimizer, self.seed_iterator)
                    training_logs.append(log)

                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, kge_model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3

                if step > 100 and step % self.p.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                    early_stopping(metrics['loss'])
                    KGEModel.log_metrics(self.p, 'Training average', step, metrics)
                    training_logs = []

                if early_stopping.early_stop:
                    break

        self.entity_embedding = kge_model.entity_embedding.detach().cpu().numpy()
        self.sub_embedding = self.entity_embedding[self.sub_index_list]
        self.relation_embedding = kge_model.relation_embedding.detach().cpu().numpy()

        return self.entity_embedding, self.sub_embedding, self.relation_embedding
