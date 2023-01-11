import os
import copy
import torch
from torch import nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, Node2Vec
from torch_geometric.data import Data
from functools import reduce

import pickle
from utils.parsers import parser
from dataLoader import read_data, Options


def init_seeds(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

opt = parser.parse_args()

######user preference parameters
bpr_lr = 0.0001
bpr_batch_size = 2048
bpr_epoch = 150

'''Learn friendship network'''
class GraphNN(nn.Module):
    def __init__(self, u_size, i_size, d_size, dropout=0.2, is_norm=True):
        super(GraphNN, self).__init__()

        self.num_users = u_size
        self.num_items = i_size
        self.ninp = d_size
        self.u_emb = nn.Embedding(u_size, self.ninp, padding_idx=0)
        self.i_emb = nn.Embedding(i_size, self.ninp, padding_idx=0)

        self.gnn1 = GCNConv(self.ninp, self.ninp)
        self.gnn2 = GCNConv(self.ninp, self.ninp)

        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)

        if self.is_norm:
            self.batch_norm = nn.LayerNorm(self.ninp)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.u_emb.weight, gain=1)
        init.xavier_uniform_(self.i_emb.weight, gain=1)
        init.xavier_uniform_(self.gnn1.lin.weight, gain=1)
        init.xavier_uniform_(self.gnn2.lin.weight, gain=1)

    def get_weight(self):
        return self.u_emb.weight, self.i_emb.weight

    def forward(self, graph):

        graph_edge_index = graph.coalesce().indices()

        users_emb = self.u_emb.weight
        items_emb = self.i_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        graph_x_embeddings = self.gnn1(all_emb, graph_edge_index)
        embs.append(graph_x_embeddings)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        embs.append(graph_output)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        # if self.is_norm:
        #     light_out = self.batch_norm(light_out)
        return light_out

def UIGraph(cascades, user_size):
    '''return the adj.'''
    e_size = len(cascades)
    n_size = user_size
    rows = []
    cols = []
    weight = []

    total_rows = []
    total_cols = []
    total_weight = []

    for i in range(e_size):
        rows += cascades[i][:-1]
        cols += [i + user_size] * (len(cascades[i]) - 1)
        weight += [1] * (len(cascades[i]) - 1)

    total_rows += rows
    total_cols += cols
    total_weight += weight

    total_rows += cols
    total_cols += rows
    total_weight += weight

    uigraph = torch.sparse_coo_tensor(torch.Tensor([rows, cols]), torch.Tensor(weight), [n_size, n_size + e_size])
    whole_graph = torch.sparse_coo_tensor(torch.Tensor([total_rows, total_cols]), torch.Tensor(total_weight), [n_size + e_size, n_size + e_size])

    return uigraph, whole_graph, e_size, len(rows)

def getUserPosItems(users, graph):
    posItems = []
    for user in users:
        posItems.append(graph[user].coalesce().indices().tolist()[0])
    return posItems

def UniformSample_original(u_size, i_size, train_size, graph):

    user_num = train_size
    users = np.random.randint(2, u_size, user_num)
    allPos = getUserPosItems(list(range(0, u_size)), graph)
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, i_size)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

##### modeling user preference
def user_preference(opt):
    _, _, _, user_size, total_cascades, _ = read_data(opt.data_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    uigraph, whole_graph, item_size, train_size = UIGraph(total_cascades, user_size)

    gnn_model = GraphNN(user_size, item_size, opt.m_pref_size, opt.dropout)
    gnn_model = gnn_model.to(device)
    optimiser = torch.optim.Adam(gnn_model.parameters(), lr=bpr_lr)

    for epoch in range(bpr_epoch):

        print('======================')
        print(f'EPOCH[{epoch}/{bpr_epoch}]-lr{bpr_lr}')

        S = UniformSample_original(user_size, item_size, train_size, uigraph)

        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        users = users.to(device)
        posItems = posItems.to(device)
        negItems = negItems.to(device)

        users, posItems, negItems = shuffle(users, posItems, negItems)
        total_batch = len(users) // bpr_batch_size + 1
        aver_loss = 0.

        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(minibatch(users, posItems, negItems, batch_size=bpr_batch_size)):

            gnn_model.train()
            optimiser.zero_grad()

            embeddings = gnn_model(whole_graph.cuda())

            users_emb = embeddings[batch_users]
            pos_emb = embeddings[batch_pos]
            neg_emb = embeddings[batch_neg]

            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)

            loss = torch.mean(F.softplus(neg_scores - pos_scores))

            loss.backward()
            optimiser.step()
            aver_loss += loss

        aver_loss = aver_loss / total_batch
        print(f'[epoch][{epoch}][saved_loss][{aver_loss}]')

    gnn_model.eval()
    embeddings = gnn_model(whole_graph.cuda())
    u_emb, i_emb = torch.split(embeddings, [user_size, item_size])

    files1 = 'data/' + opt.data_name + '/uemb'+ str(opt.m_pref_size) +'.pickle'
    with open(files1, 'wb') as handle:
        pickle.dump(u_emb.cpu().detach().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    files2 = 'data/' + opt.data_name + '/iemb'+ str(opt.m_pref_size) +'.pickle'
    with open(files2, 'wb') as handle:
        pickle.dump(i_emb.cpu().detach().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

##### Node2vec
def node2vec_function(opt):
    data = opt.data_name
    node2vec_dim = opt.m_struc_size

    files = 'data/' + data + '/node2vec'+ str(opt.m_struc_size)+ '.pickle'

    options = Options(data)
    _u2idx = {}

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    edges_list = []
    if os.path.exists(options.net_data):
        with open(options.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]

            edges = copy.deepcopy(relation_list)
            result = reduce(lambda x, y: x.extend(y) or x, edges)

            keys = list(_u2idx.keys())
            result = result + keys
            uniq_result = list(set(result))
            uidx = [i for i in range(len(uniq_result))]
            u2idx = dict(zip(uniq_result, uidx))
            node_num = len(u2idx)

            relation_list = [(u2idx[edge[0]], u2idx[edge[1]]) for edge in relation_list]

            relation_list_reverse = [edge[::-1] for edge in relation_list]
            edges_list += relation_list_reverse
    else:
        return []
    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()

    data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=node2vec_dim, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True, num_nodes=node_num).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, 81):
        loss = train()
        print("epoch:", epoch)
        print("loss", loss)

    z = model()
    keys = list(_u2idx.keys())
    idxs = [u2idx[i] for i in keys]
    idxs = torch.LongTensor(idxs)
    node_emb = F.embedding(idxs.cuda(), z.cuda())

    with open(files, 'wb') as handle:
        pickle.dump(node_emb.cpu().detach().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return node_emb.cpu().detach().numpy()



if __name__ == "__main__":

    user_preference(opt)

    node2vec_function(opt)


