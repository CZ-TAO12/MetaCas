import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class MetaGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, hyper_size, meta_size, heads=1, concat=True, negative_slope=0.2, dropout=0.3, bias=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MetaGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.__alpha__ = None

        self.att_i_o = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_j_o = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))

        self.input_size = meta_size + in_channels

        self.hidden_size = hyper_size

        self.att_size = self.heads * self.out_channels
        self.W_size = self.in_channels * self.heads * self.out_channels

        self.lin_f_hidden = nn.Linear(self.input_size, self.hidden_size, bias=False)

        self.W_hyper = nn.Linear(self.hidden_size, self.W_size, bias=False)

        self.att_hyper = nn.Linear(self.hidden_size, 2 * self.att_size, bias=False)

        self.layer_norm = nn.LayerNorm(self.hidden_size)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        nn.init.xavier_uniform_(self.lin_f_hidden.weight)
        nn.init.xavier_uniform_(self.W_hyper.weight)
        nn.init.xavier_uniform_(self.att_hyper.weight)
        nn.init.xavier_uniform_(self.att_i_o)
        nn.init.xavier_uniform_(self.att_j_o)

    def forward(self, x, meta_x, edge_index, return_attention_weights=False):

        meta_inp = torch.cat([x, meta_x], dim=-1)

        f_out = self.lin_f_hidden(meta_inp)
        f_out = torch.tanh(f_out)

        w_hyper = self.W_hyper(f_out)
        W_hyper = w_hyper.view(-1, self.heads * self.out_channels, self.in_channels)

        x = torch.bmm(x.unsqueeze(1), W_hyper.transpose(1, 2)).squeeze(1)

        att_hyper = self.att_hyper(f_out)
        att_i, att_j = att_hyper.chunk(2, dim=-1)
        self.att_i = att_i.view(-1, self.heads, self.out_channels)
        self.att_j = att_i.view(-1, self.heads, self.out_channels)

        x = (x, x)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.contiguous().view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, edge_index_j, size_i,
                return_attention_weights):
        # Compute attention coefficients.
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i_o * self.att_i[edge_index_i]).sum(-1) + (x_j * self.att_j_o * self.att_j[edge_index_j]).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index = edge_index_i, num_nodes = size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        ####TODO: attente this line
        output = x_j * alpha.view(-1, self.heads, 1)

        return output

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MetaGNN(nn.Module):
    def __init__(self, models_name, ntoken, ninp, nout,  hyper_size=128, meta_size = 64, dropout=0.3):
        super(MetaGNN, self).__init__()

        self.u_size = ntoken
        self.ninp = ninp
        self.nout = nout
        self.hyper_size = hyper_size
        self.meta_size = meta_size
        self.models_name = models_name

        ###node embedding
        self.node_embedding = nn.Embedding(self.u_size, self.ninp, padding_idx=0)
        ###build GAT model
        if models_name=='gat':
            self.model_layers = MetaGATConv(self.ninp, self.nout, self.hyper_size, self.meta_size)
        else:
            print(0)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)

    @staticmethod
    def create_mask(x, p, training):
        device = x.device
        mask = torch.ones_like(x, dtype=torch.float, device=device)
        if training:
            mask = (torch.rand(*x.shape, dtype=torch.float, device=device) > p) / (1 - p)
        return mask

    def create_models_params(self, x, meta_k, edge_index, edge_weight):
        if self.models_name == 'gat':
            return F.elu, [x, meta_k, edge_index]
        else:
            print("model name is wrong")


    def forward(self, data, meta_k):

        x = self.node_embedding.weight
        edge_index, edge_weight = data.edge_index, data.edge_attr
        mask = self.create_mask(x, p=self.dropout, training=self.training)
        x = x * mask

        model_act, model_params = self.create_models_params(x, meta_k, edge_index, edge_weight)

        out = model_act(self.model_layers(*model_params))

        return out

