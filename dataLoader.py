import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle

class Options(object):
    def __init__(self, data_name='twitter'):
        self.data = 'data/' + data_name + '/cascades.txt'
        self.u2idx_dict = 'data/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = 'data/' + data_name + '/idx2u.pickle'
        self.save_path = ''
        self.net_data = 'data/' + data_name + '/edges.txt'

        # train file path.
        self.train_data = 'data/' + data_name + '/cascadetrain.txt'
        # valid file path.
        self.valid_data = 'data/' + data_name + '/cascadevalid.txt'
        # test file path.
        self.test_data = 'data/' + data_name + '/cascadetest.txt'

def _readFromFile(filename, with_EOS):
    t_cascades = []
    timestamps = []
    cas_idx = []
    ####process the raw data
    for line in open(filename):
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []
        chunks = line.strip().split()
        for i, chunk in enumerate(chunks):
            if i ==0:
                cas_idx.append(int(chunk))
            else:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(chunk)

                userlist.append(int(user))
                timestamplist.append(float(timestamp))

        if len(userlist) > 1:
            if with_EOS:
                userlist.append(Constants.EOS)
                timestamplist.append(timestamplist[-1])

            t_cascades.append(userlist)
            timestamps.append(timestamplist)

    return t_cascades, timestamps, cas_idx

def buildIndex(data):
    user_set = set()
    u2idx = {}
    idx2u = []

    lineid = 0
    for line in open(data):
        lineid += 1
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    user_set.add(root)
            except:
                print(line)
                print(chunk)
                print(lineid)
            user_set.add(user)
    pos = 0
    u2idx['<blank>'] = pos
    idx2u.append('<blank>')
    pos += 1
    u2idx['</s>'] = pos
    idx2u.append('</s>')
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        idx2u.append(user)
        pos += 1
    user_size = len(user_set) + 2
    print("user_size : %d" % (user_size))
    return user_size, u2idx, idx2u

def read_data(data_name, random_seed=2022, with_EOS=True):
    options = Options(data_name)

    '''user size'''
    with open(options.u2idx_dict, 'rb') as handle:
        u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        idx2u = pickle.load(handle)
    user_size = len(u2idx)

    '''load train data, validation data and test data'''
    train, train_t, train_idx  = _readFromFile(options.train_data, with_EOS)
    valid, valid_t, valid_idx = _readFromFile(options.valid_data, with_EOS)
    test, test_t, test_idx = _readFromFile(options.test_data, with_EOS)

    t_cascades = train + valid + test
    timestamps = train_t + valid_t + test_t

    '''shuffle training data'''
    random.seed(random_seed)
    random.shuffle(train)
    random.seed(random_seed)
    random.shuffle(train_t)
    random.seed(random_seed)
    random.shuffle(train_idx)

    train = [train, train_t, train_idx]
    valid = [valid, valid_t, valid_idx]
    test = [test, test_t, test_idx]

    return train, valid, test, user_size, t_cascades, timestamps

def Split_data(data_name, train_rate=0.8, valid_rate=0.1, load_dict=True):

    options = Options(data_name)

    if not load_dict:
        user_size, u2idx, idx2u = buildIndex(options.data)
        with open(options.u2idx_dict, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as handle:
            pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)
    user_size = len(u2idx)

    t_cascades = []
    timestamps = []
    ####process the raw data
    for line in open(options.data):
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []

        chunks = line.strip().strip(',').split(',')
        for chunk in chunks:
            try:
                # Twitter,Douban, memes
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                # Android
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()

                    userlist.append((u2idx[root]))
                    timestamplist.append(float(timestamp))
            except:
                print(chunk)

            userlist.append((u2idx[user]))
            timestamplist.append(float(timestamp))

        t_cascades.append(userlist)
        timestamps.append(timestamplist)

    '''ordered by timestamps'''
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    timestamps = sorted(timestamps)
    t_cascades[:] = [t_cascades[i] for i in order]
    cas_idx = [i for i in range(len(t_cascades))]

    '''data split'''
    train_idx_ = int(train_rate * len(t_cascades))
    train = t_cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]
    train_idx = cas_idx[0:train_idx_]

    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]
    valid_idx = cas_idx[train_idx_:valid_idx_]

    test = t_cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]
    test_idx = cas_idx[valid_idx_:]

    '''empty folder'''
    with open(options.train_data, 'w') as file:
        file.truncate(0)

    '''write training set '''
    with open(options.train_data, 'w') as f:
        for i in range(len(train)):
            data = list(zip(train[i], train_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            data = str(train_idx[i]) + ' ' + ' '.join(data_lst0)
            f.writelines(data + "\n")

    with open(options.valid_data, 'w') as file:
        file.truncate(0)

    '''write validation set '''
    with open(options.valid_data, 'w') as f:
        for i in range(len(valid)):
            data = list(zip(valid[i], valid_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            data = str(valid_idx[i]) + ' ' + ' '.join(data_lst0)
            f.writelines(data + "\n")

    with open(options.test_data, 'w') as file:
        file.truncate(0)

    '''write testing set '''
    with open(options.test_data, 'w') as f:
        for i in range(len(test)):
            data = list(zip(test[i], test_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            data = str(test_idx[i]) + ' ' + ' '.join(data_lst0)
            f.writelines(data + "\n")

    total_len = sum(len(i) - 1 for i in t_cascades)
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
    print("total size:%d " % (len(t_cascades)))
    print("average length:%f" % (total_len / len(t_cascades)))
    print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
    print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))
    print("user size:%d" % (user_size - 2))

    return user_size, t_cascades, timestamps

class DataLoader(object):
    ''' For data iteration '''
    def __init__(
            self, cas, batch_size=64, data_type='training', cuda=True, test=False, with_EOS=True):
        self._batch_size = batch_size
        self.cas = cas[0]
        self.time = cas[1]
        self.idx = cas[2]
        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda
        self.data_type = data_type

        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''
            max_len = 200
            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            seq_idx = Variable(
                torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)

            return seq_data, seq_data_timestamp, seq_idx
        else:
            if self.data_type == 'training':
                random.shuffle(self.cas)
            self._iter_count = 0
            raise StopIteration()
