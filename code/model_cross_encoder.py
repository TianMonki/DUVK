#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
from random import sample

import torch
import torch.nn as nn
from torch import optim
import math
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, DebertaTokenizer, BertConfig, \
    DebertaModel, AutoTokenizer, AutoModel, \
    AutoModelForSequenceClassification
from utils import EarlyStopping
from torch.nn.functional import cosine_similarity


class BertModelCross(nn.Module):
    def __init__(self, max_length, target_dim):
        super(BertModelCross, self).__init__()
        self.max_length = max_length
        self.target_dim = target_dim
        self.tokenizer = BertTokenizer.from_pretrained('../plm/unsup-simcse-bert-base-uncased')
        self.bert = BertModel.from_pretrained('../plm/unsup-simcse-bert-base-uncased')
        self.tokenizer.add_tokens(['[TRI]'])
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768, target_dim)

    def forward(self, sample):
        batch_tokenized = self.tokenizer.batch_encode_plus(sample, add_special_tokens=True,
                                                           max_length=self.max_length,
                                                           pad_to_max_length=True)

        for i in range(len(batch_tokenized['input_ids'])):
            if i % 2 == 1:
                batch_tokenized['input_ids'][i].pop(0)
                batch_tokenized['attention_mask'][i].pop(0)
                batch_tokenized['input_ids'][i].append(0)
                batch_tokenized['attention_mask'][i].append(0)

        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        input_ids = input_ids.view(int(input_ids.size()[0] / 2), self.max_length * 2)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        attention_mask = attention_mask.view(int(attention_mask.shape[0] / 2), self.max_length * 2)

        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[1]
        linear_output = self.dense(bert_cls_hidden_state)

        return linear_output


class Bert_Model_Cross(object):
    def __init__(self, params, side_info, intersection_seed, intersection_cosines, union_seed):
        self.p = params
        self.side_info = side_info
        self.intersection_seed = intersection_seed
        self.intersection_cosines = intersection_cosines
        self.batch_size = 2
        self.negative_sample_size = 5
        self.union_seed = union_seed

        # self.negative_sample_size = 5

        self.epochs = 10
        # self.lr = 0.005
        self.lr = 0.01
        # self.K = K
        print('self.epochs:', self.epochs)
        self.max_length = 256
        self.bert_embedding_model = BertModelCross(self.max_length, target_dim=1).cuda()
        self.sigm = nn.Sigmoid()

    def fine_tune(self):

        pair_samples, pair_cosines = [], []
        for i in range(len(self.intersection_seed)):
            sub1_sequence = self.side_info.sub2triple_sequence[self.side_info.id2sub[self.intersection_seed[i][0]]]
            sub2_sequence = self.side_info.sub2triple_sequence[self.side_info.id2sub[self.intersection_seed[i][1]]]
            pair_samples.append(sub1_sequence)
            pair_samples.append(sub2_sequence)
            pair_samples.append(sub2_sequence)
            pair_samples.append(sub1_sequence)
            pair_cosines.append(self.intersection_cosines[self.intersection_seed[i]])

            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < self.negative_sample_size:
                negative_sample = random.randint(0, len(self.side_info.sub_list) - 1)
                if negative_sample_size % 2 == 1:
                    if negative_sample > self.intersection_seed[i][0]:
                        if (self.intersection_seed[i][0], negative_sample) not in self.union_seed:
                            negative_sample_list.append((self.intersection_seed[i][0], negative_sample))
                            negative_sample_size += 1
                    elif self.intersection_seed[i][0] > negative_sample:
                        if (negative_sample, self.intersection_seed[i][0]) not in self.union_seed:
                            negative_sample_list.append((negative_sample, self.intersection_seed[i][0]))
                            negative_sample_size += 1
                else:
                    if negative_sample > self.intersection_seed[i][1]:
                        if (self.intersection_seed[i][1], negative_sample) not in self.union_seed:
                            negative_sample_list.append((self.intersection_seed[i][1], negative_sample))
                            negative_sample_size += 1
                    elif self.intersection_seed[i][1] > negative_sample:
                        if (negative_sample, self.intersection_seed[i][1]) not in self.union_seed:
                            negative_sample_list.append((negative_sample, self.intersection_seed[i][1]))
                            negative_sample_size += 1

            for j in range(len(negative_sample_list)):
                np0 = negative_sample_list[j][0]
                np1 = negative_sample_list[j][1]
                sub0 = self.side_info.id2sub[np0]
                sub1 = self.side_info.id2sub[np1]
                sub0_seq = self.side_info.sub2triple_sequence[sub0]
                sub1_seq = self.side_info.sub2triple_sequence[sub1]
                pair_samples.append(sub0_seq)
                pair_samples.append(sub1_seq)
                pair_samples.append(sub1_seq)
                pair_samples.append(sub0_seq)
                pair_cosines.append(float(0))

        self.batch_count = math.ceil(len(self.intersection_seed) / self.batch_size)
        print('batch_count:', self.batch_count)

        self.batch_train_inputs, self.batch_train_targets = [], []
        for i in range(self.batch_count):
            self.batch_train_inputs.append(pair_samples[i * self.batch_size * (1 + self.negative_sample_size) * 2 * 2:
                                                        (i + 1) * self.batch_size * (
                                                                    1 + self.negative_sample_size) * 2 * 2])
            self.batch_train_targets.append(pair_cosines[i * self.batch_size * (1 + self.negative_sample_size): (
                                                                                                                            i + 1) * self.batch_size * (
                                                                                                                            1 + self.negative_sample_size)])

        optimizer = optim.SGD(self.bert_embedding_model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=3, delta=0.1)

        for epoch in range(self.epochs):
            avg_epoch_loss = 0
            for i in range(self.batch_count):
                inputs = self.batch_train_inputs[i]
                batch_len = int(len(inputs) / ((1 + self.negative_sample_size) * 4))
                labels = torch.tensor(self.batch_train_targets[i]).cuda().view(batch_len,
                                                                               (1 + self.negative_sample_size))
                optimizer.zero_grad()

                linear_output = self.bert_embedding_model(inputs)
                linear_output = linear_output.view(batch_len, (1 + self.negative_sample_size), 2)
                linear_output = torch.sum(linear_output, dim=2) / 2
                loss = criterion(linear_output, labels)

                # early-stop
                loss.backward()
                nn.utils.clip_grad_norm_(self.bert_embedding_model.parameters(), 1.0)
                optimizer.step()
                avg_epoch_loss += loss.item()
                if i == (self.batch_count - 1):
                    real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                    print(real_time, "Epoch: %d, Loss: %.4f" % (epoch, avg_epoch_loss))
            early_stopping(avg_epoch_loss)
            if early_stopping.early_stop:
                break

        torch.save(self.bert_embedding_model.state_dict(), '../output/state_dict/multi_view/' + self.p.dataset+ '/crossencoder.pth')

    def load_state(self):
        self.bert_embedding_model.load_state_dict(torch.load('../output/state_dict/multi_view/' + self.p.dataset+ '/crossencoder.pth'))

    def inference(self):
        pair_samples, pair_cosines = [], []
        for i in range(len(self.union_seed)):
            sub1_sequence = self.side_info.sub2triple_sequence[self.side_info.id2sub[self.union_seed[i][0]]]
            sub2_sequence = self.side_info.sub2triple_sequence[self.side_info.id2sub[self.union_seed[i][1]]]
            pair_samples.append(sub1_sequence)
            pair_samples.append(sub2_sequence)
            pair_samples.append(sub2_sequence)
            pair_samples.append(sub1_sequence)

        batch_count = math.ceil(len(self.union_seed) / self.batch_size)
        print('batch_count:', batch_count)

        batch_train_inputs, batch_train_targets = [], []
        for i in range(batch_count):
            batch_train_inputs.append(pair_samples[i * self.batch_size * 2 * 2:
                                                   (i + 1) * self.batch_size * 2 * 2])
            batch_train_targets.append(pair_cosines[i * self.batch_size:
                                                    (i + 1) * self.batch_size])

        with torch.no_grad():
            # scores = []
            pair_score = dict()
            for i in range(batch_count):
                inputs = batch_train_inputs[i]
                batch_len = int(len(inputs) / 4)
                linear_output = self.bert_embedding_model(inputs)
                sigmoid_score = self.sigm(linear_output.view(batch_len, 2))
                sigmoid_score = sigmoid_score.view(batch_len, 2)
                sigmoid_score = (torch.sum(sigmoid_score, dim=1) / 2).cpu()
                for j in range(len(sigmoid_score)):
                    pair_score[self.union_seed[i * self.batch_size + j]] = sigmoid_score[j].item()

        return pair_score
