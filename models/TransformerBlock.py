import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, episilon=1e-6):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / (self.temperature + episilon)

        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, k.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
            mask_ = mask + pad_mask
            attn = attn.masked_fill(mask_, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, n_head, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        X = self.layer_norm(output + residual)
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        output = self.layer_norm(output + X)

        return output, attn

class AttnModel(torch.nn.Module):
    """Attention
    """
    def __init__(self, feat_dim, meta_dim, out_dim, n_head=8, drop_out=0.2):
        """
        args:
          feat_dim: dim for the user features
          meta_dim: dim for the meta features
          out_dim: dim for the output
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.meta_dim = meta_dim
        # self.out_dim = out_dim

        self.input_dim = (feat_dim + meta_dim)
        self.model_dim = self.input_dim
        self.out_dim = self.model_dim

        assert (self.out_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)

        self.multi_head_target = MultiHeadAttention(n_head,
                                                    d_model=self.model_dim,
                                                    d_k=self.out_dim // n_head,
                                                    d_v=self.out_dim // n_head,
                                                    dropout=drop_out)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, seq_x, seq_meta, mask):


        seq_x = seq_x.permute(1, 0, 2)
        seq_meta = seq_meta.permute(1, 0, 2)

        q = torch.cat([seq_x, seq_meta], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq_x, seq_meta], dim=2)   # [B, 1, D + De + Dt] -> [B, 1, D]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]

        output = output.squeeze()
        attn = attn.squeeze()

        output = output.permute(1, 0, 2)
        attn = attn.permute(1, 0, 2)

        return output, attn