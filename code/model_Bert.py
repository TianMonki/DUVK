#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from random import sample

import torch
import torch.nn as nn
from torch import optim
import math
from helper import *
import json
import logging
import os
import gensim
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, DebertaTokenizer, BertConfig, DebertaModel, AutoTokenizer, AutoModel, \
    AutoModelForSequenceClassification
from utils import EarlyStopping
from torch.nn.functional import cosine_similarity


def get_true_head_and_tail(pairs):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''

    true_head = {}
    true_tail = {}

    for head, tail in pairs:
        if head not in true_tail:
            true_tail[head] = []
        true_tail[head].append(tail)
        if tail not in true_head:
            true_head[tail] = []
        true_head[tail].append(head)

    for tail in true_head:
        true_head[tail] = np.array(list(set(true_head[tail])))
    for head in true_tail:
        true_tail[head] = np.array(list(set(true_tail[head])))

    return true_head, true_tail


class BertEmbeddingModel(nn.Module):
    def __init__(self, max_length, target_dim):
        super(BertEmbeddingModel, self).__init__()
        self.max_length = max_length
        self.target_dim = target_dim
        self.tokenizer = BertTokenizer.from_pretrained('../plm/unsup-simcse-bert-base-uncased')
        self.bert = BertModel.from_pretrained('../plm/unsup-simcse-bert-base-uncased')
        self.tokenizer.add_tokens(['[TRI]'])
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.dense = nn.Linear(768, target_dim)

    def forward(self, sample):
        batch_tokenized = self.tokenizer.batch_encode_plus(sample, add_special_tokens=True,
                                                           max_length=self.max_length,
                                                           pad_to_max_length=True)
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        linear_output = self.dense(bert_cls_hidden_state)

        return bert_cls_hidden_state, linear_output


class Bert_Model(object):
    def __init__(self, params, side_info):
        self.p = params
        self.side_info = side_info
        self.batch_size = 6
        self.negative_sample_size = 5
        self.epochs = 200
        # self.lr = 0.005
        self.lr = 0.001
        # self.K = K
        print('self.epochs:', self.epochs)
        self.max_length = 256
        self.bert_embedding_model = BertEmbeddingModel(self.max_length, target_dim=300).cuda()

    def encode(self):
        linear_outputs = dict()

        train_sequences, train_subs = [], []
        for sub in self.side_info.sub_list:
            sub_sequence = self.side_info.sub2triple_sequence[sub]
            train_sequences.append(sub_sequence)
            train_subs.append(sub)
        # train_sequences = np.array(train_sequences)
        # train_subs = np.array(train_subs)

        batch_count = math.ceil(len(self.side_info.sub_list) / self.batch_size)
        print('batch_count:', batch_count)

        batch_sub_sequence = []
        batch_sub = []
        for i in range(batch_count):
            batch_sub_sequence.append(train_sequences[i * self.batch_size:(i + 1) * self.batch_size])
            batch_sub.append(train_subs[i * self.batch_size:(i + 1) * self.batch_size])

        with torch.no_grad():
            for i in range(batch_count):
                bert_cls_hidden_state, linear_output = self.bert_embedding_model(batch_sub_sequence[i])
                for si in range(len(batch_sub[i])):
                    linear_outputs[batch_sub[i][si]] = (linear_output.cpu().numpy())[si]

        print("Primary Context View Embedding")
        return linear_outputs

    def load_state(self):
        self.bert_embedding_model.load_state_dict(
            torch.load('../output/state_dict/multi_view/' + self.p.dataset + '/biencoder.pth'))

    def fine_tune(self, input_list, seed_pairs, topks):
        self.seed_pairs = seed_pairs
        self.true_head, self.true_tail = get_true_head_and_tail(self.seed_pairs)

        sub_sequence_samples = []
        for seed_pair in self.seed_pairs:
            # sub_sequence_sample = []
            sub_sequence = self.side_info.sub2triple_sequence[input_list[seed_pair[0]]]
            sub_pos_sequence = self.side_info.sub2triple_sequence[input_list[seed_pair[1]]]
            sub_sequence_samples.append(sub_sequence)
            sub_sequence_samples.append(sub_pos_sequence)

            sub_topk = topks[seed_pair[0]]
            sub_topk.append(seed_pair[0])
            # intersection negative-sample
            negative_sample_list = []
            negative_sample_size = 0
            while negative_sample_size < self.negative_sample_size:
                negative_sample = sample(
                    [self.side_info.ent2id[j] for j in self.side_info.sub_list if
                     self.side_info.ent2id[j] not in sub_topk], 1)
                if (input_list[seed_pair[0]], negative_sample[0]) not in self.seed_pairs:
                    negative_sample_list.append(negative_sample[0])
                    negative_sample_size += 1
                else:
                    continue
            for j in negative_sample_list:
                sub = self.side_info.id2sub[j]
                sub_sequence_samples.append(self.side_info.sub2triple_sequence[sub])
            # sub_sequence_samples.append(sub_sequence_sample)

        batch_count = math.ceil(len(self.seed_pairs) / self.batch_size)
        print('batch_count:', batch_count)

        batch_train_inputs, batch_train_targets = [], []
        for i in range(batch_count):
            batch_train_inputs.append(sub_sequence_samples[i * self.batch_size * (2 + self.negative_sample_size):
                                                           (i + 1) * self.batch_size * (2 + self.negative_sample_size)])

        optimizer = optim.SGD(self.bert_embedding_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=3, delta=5)

        for epoch in range(self.epochs):
            avg_epoch_loss = 0
            for i in range(batch_count):
                inputs = batch_train_inputs[i]
                batch_len = int(len(inputs) / (2 + self.negative_sample_size))
                labels = torch.zeros(batch_len, dtype=torch.int64).cuda()
                self.bert_cls_hidden_state, outputs = self.bert_embedding_model(inputs)
                outputs = outputs.view(batch_len, (2 + self.negative_sample_size), 300)
                outputs_pos = outputs[:, 0, :].unsqueeze(2)
                outputs_neg = outputs[:, 1:, :]
                logits = torch.matmul(outputs_neg, outputs_pos).squeeze(-1)

                loss = criterion(logits, labels)
                # early-stop

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.bert_embedding_model.parameters(), 1.0)
                optimizer.step()
                avg_epoch_loss += loss.item()
                if i == (batch_count - 1):
                    real_time = time.strftime("%Y_%m_%d") + ' ' + time.strftime("%H:%M:%S")
                    print(real_time, "Epoch: %d, Loss: %.4f" % (epoch, avg_epoch_loss))

            early_stopping(avg_epoch_loss)
            if early_stopping.early_stop:
                break

        torch.save(self.bert_embedding_model.state_dict(),
                   '../output/state_dict/multi_view/' + self.p.dataset + '/biencoder.pth')

        sub_sequences = []
        for i in input_list:
            sub_sequences.append(self.side_info.sub2triple_sequence[i])
        batch_embedding_inputs = []
        batch_cnt = math.ceil(len(input_list) / self.batch_size)

        for i in range(batch_cnt):
            batch_embedding_inputs.append(sub_sequences[i * self.batch_size:(i + 1) * self.batch_size])

        with torch.no_grad():
            bert_embedding = []
            for i in range(batch_cnt):
                input = batch_embedding_inputs[i]
                self.bert_cls_hidden_state, outputs = self.bert_embedding_model(input)
                for op in outputs:
                    bert_embedding.append(op.cpu().numpy())

        print('Fine-tune Bert Embedding')
        return bert_embedding
