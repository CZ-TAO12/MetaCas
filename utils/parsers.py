import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_name', type=str, default='android', choices=['android', 'memes', 'twitter', 'douban'], help="dataset")
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=72)
parser.add_argument('-pos_emb', type=bool, default=True)

parser.add_argument('-m_struc_size', type=int, default=64, help="the dimension of node feature in Node2vec")
parser.add_argument('-m_pref_size', type=int, default=64, help="the dimension of user preference in GNN")
parser.add_argument('-m_time_size', type=int, default=32, help="the dimension of time feature")
##### model parameters
parser.add_argument('-d_model', type=int, default=64, help="the dimension of model")
parser.add_argument('-initialFeatureSize', type=int, default=64, help="the dimension of user embedding")

parser.add_argument('-meta_v', type=int, default=128)
parser.add_argument('-meta_d', type=int, default=64)

parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.2)

parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)
parser.add_argument('-log', default=None)


parser.add_argument('-save_path', default= "./checkpoint/")
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
parser.add_argument('-patience', type=int, default=5, help="control the step of early-stopping")

parser.add_argument('-split_data', type=bool, default=False, help="control data preprocess: split_data")
parser.add_argument('-gnn_type', type=str, choices=['gat'], default='gat')

