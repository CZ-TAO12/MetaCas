import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.Meta_GNN import MetaGNN
from models.MetaLSTM import MetaLSTM

'''Mask previous activated users'''
def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.temperature = np.power(time_dim, 0.5)
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        self.layer_norm = nn.LayerNorm(time_dim)

    def forward(self, ts):
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ft = ts[:,0].view(batch_size, 1)
        ts = ft - ts

        ts = ts.view(batch_size, seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)

        output = torch.cos(map_ts)
        return output

class MetaCas(nn.Module):
    def __init__(self, opt, dropout=0.3):
        super(MetaCas, self).__init__()
        self.gnn_type = opt.gnn_type
        ###
        self.dropout = nn.Dropout(dropout)

        self.pos_dim = 8
        self.pos_embedding = nn.Embedding(500, self.pos_dim)
        self.hidden_size = opt.d_word_vec
        self.n_node = opt.user_size
        self.input_dim = opt.initialFeatureSize

        self.meta_v = opt.meta_v
        self.gnn_meta_dim = opt.m_struc_size + opt.m_pref_size

        self.gnn = MetaGNN(self.gnn_type, self.n_node, self.input_dim, self.hidden_size, self.meta_v, self.gnn_meta_dim, dropout=dropout)

        self.time_dim = opt.m_time_size
        self.lstm_m_dim = self.gnn_meta_dim + self.time_dim

        self.meta_d = opt.meta_d


        self.rnn = MetaLSTM(self.hidden_size+ self.pos_dim, self.hidden_size+ self.pos_dim, self.lstm_m_dim, self.meta_d, self.meta_d)
        self.time_encoder = TimeEncode(self.time_dim)

        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, input_timestamp, graph, meta):

        devices = input.device
        input = input[:, :-1]
        input_timestamp = input_timestamp[:, :-1]

        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))

        meta_s = torch.from_numpy(meta).to(torch.float32).to(devices)


        g_hidden = self.gnn(graph, meta_s)

        dyemb = F.embedding(input.cuda(), g_hidden)
        dyemb = torch.cat([dyemb, order_embed], dim=-1).cuda()
        dyemb = dyemb.permute(1, 0, 2)

        meta_time = self.time_encoder(input_timestamp)
        meta_seq = F.embedding(input, meta_s)
        meta_t = torch.cat([meta_seq, meta_time], -1)
        meta_t = meta_t.permute(1, 0, 2)
        hidden1, state1 = self.rnn(dyemb, meta_t, input)

        att_out = self.dropout(hidden1)
        out = att_out.permute(1, 0, 2)

        # conbine users and cascades
        output_u = self.linear2(out.cuda())  # (bsz, user_len, |U|)
        mask = get_previous_user_mask(input.cpu(), self.n_node)

        return (output_u + mask).view(-1, output_u.size(-1)).cuda()


