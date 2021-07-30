"""
@Time   :   2021-07-28 17:34:56
@File   :   modeling_cdnet.py
@Author :   okcd00
@Email  :   okcd00{at}qq.com
"""

import os
import operator
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
import pytorch_lightning as pl


class DetectionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True,
        )

        self.context_lstm = nn.GRU(  # or nn.LSTM
            input_size=config.hidden_size,  # 768
            hidden_size=config.hidden_size // 4,  # 192
            num_layers=1,
            bidirectional=True)

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.config.hidden_size, 1)

    def context_hidden(self, input_ids, token_type_ids=None, attention_mask=None):
        # [batch, sequence_length + 1], in case of right overflow
        am = attention_mask
        inp_ids = self.add_pad(input_ids)
        sequence_length = am.sum(-1)

        # ([batch], [batch])
        sent_indexes = torch.arange(am.shape[0], device=am.device)
        char_indexes = sequence_length.long().to(device=am.device)
        indexes = (sent_indexes, char_indexes)
        input_ids_with_sep = inp_ids.index_put_(
            indexes, torch.tensor(102, device=am.device))

        # [batch, sequence_length w/ [CLS] [SEP], embedding_size]
        embeddings = self.bert.embeddings(
            input_ids=input_ids_with_sep,  # self.add_sep(input_ids),
            token_type_ids=token_type_ids)

        # the LSTM part
        packed = pack(embeddings, sequence_length + 1,  # add [SEP]
                      batch_first=True, enforce_sorted=False)
        token_hidden, (h_n, c_n) = self.context_lstm(packed)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = unpack(token_hidden, batch_first=True)[0]

        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = token_hidden.view(token_hidden.shape[0], token_hidden.shape[1], 2, -1)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size] * 2 directions
        return token_hidden[:, :, 0], token_hidden[:, :, 1]

    def forward(self, hidden_states):
        out, _ = self.gru(hidden_states)
        prob = self.linear(out)
        prob = self.sigmoid(prob)
        return prob


