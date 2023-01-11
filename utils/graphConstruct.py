import torch
import pickle
import os
from torch_geometric.data import Data
from dataLoader import Options

'''Friendship network'''
def ConRelationGraph(data):
    options = Options(data)
    _u2idx = {}

    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    edges_list = []
    if os.path.exists(options.net_data):
        with open(options.net_data, 'r') as handle:
            relation_list = handle.read().strip().split("\n")
            relation_list = [edge.split(',') for edge in relation_list]

            relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                             edge[0] in _u2idx and edge[1] in _u2idx]
            relation_list_reverse = [edge[::-1] for edge in relation_list]
            edges_list += relation_list_reverse
    else:
        return []
    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()
    data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return data


