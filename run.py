import os
import time
import pickle
import Constants
from tqdm import tqdm
from utils.graphConstruct import ConRelationGraph
from dataLoader import read_data, DataLoader, Split_data
from models.models import *
from Optim import ScheduledOptim
from utils.EarlyStopping import *

from utils.parsers import parser
from utils.Metrics import Metrics

def init_seeds(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

metric = Metrics()
opt = parser.parse_args() 
opt.d_word_vec = opt.d_model

def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct

def train_epoch(model, training_data, graph, meta, loss_func, optimizer):
    # train
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    batch_num = 0.0

    for  batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # data preparing
        tgt, tgt_timestamp, tgt_idx = (item.cuda() for item in batch)
        gold = tgt[:, 1:]
        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)
        
        # training
        optimizer.zero_grad()
        pred = model(tgt, tgt_timestamp, graph, meta)
        
        # loss
        loss, n_correct = get_performance(loss_func, pred, gold)
        loss.backward()

        # parameter update
        optimizer.step()
        optimizer.update_learning_rate()

        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss/n_total_words, n_total_correct/n_total_words

def train_model(data_path, meta):
    # ========= Preparing DataLoader =========#
    train, valid, test, user_size, total_cascades, timestamps = read_data(data_path)

    train_data = DataLoader(train, batch_size=opt.batch_size, data_type='training', cuda=False)
    valid_data = DataLoader(valid, batch_size=opt.batch_size, data_type='validing', cuda=False)
    test_data = DataLoader(test, batch_size=opt.batch_size, data_type='testing', cuda=False)

    relation_graph = ConRelationGraph(data_path).cuda()
    opt.user_size = user_size

    # ========= Preparing Model =========#
    model = MetaCas(opt, dropout=opt.dropout)

    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    
    params = model.parameters()
    optimizerAdam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    # ========= early-stopping =========#
    save_model_path = opt.save_path + opt.data_name + '_' + "MetaGAT" '_' + "MetaLSTM" + '.pt'
    if not os.path.exists(save_model_path):
        os.system(r"touch {}".format(save_model_path))
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path= save_model_path)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    for epoch_i in range(opt.epoch):
        print('\n[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, relation_graph, meta,  loss_func, optimizer)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        if epoch_i >= 1:
            start = time.time()
            val_scores = epoch_test(model, valid_data, relation_graph, meta)
            print('  - ( Validation ) ')
            for metric in val_scores.keys():
                print(metric + ' ' + str(val_scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")

            print('  - (Test) ')
            test_scores = epoch_test(model, test_data, relation_graph, meta)
            for metric in test_scores.keys():
                print(metric + ' ' + str(test_scores[metric]))

            early_stopping(-sum(list(val_scores.values())[-2:]), model)

            if early_stopping.early_stop:
                print("Early_Stopping")
                break

    model_dict = torch.load(save_model_path)
    model.load_state_dict(model_dict)
    print(" -(Finished!!) \n best scores: ")
    test_scores = epoch_test(model, test_data, relation_graph, meta)
    for metric in test_scores.keys():
        print(metric + ' ' + str(test_scores[metric]))

def epoch_test(model, validation_data, graph, meta, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            # prepare data
            tgt, tgt_timestamp, tgt_idx = (item.cuda() for item in batch)

            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            # forward
            pred = model(tgt, tgt_timestamp, graph, meta)
            y_pred = pred.detach().cpu().numpy()

            scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores

if __name__ == "__main__":
    init_seeds(seed=2022)

    files = 'data/' + opt.data_name + '/node2vec' + str(opt.m_struc_size) + '.pickle'
    files1 = 'data/' + opt.data_name + '/uemb' + str(opt.m_pref_size) + '.pickle'

    if opt.split_data:
        Split_data(opt.data_name, train_rate=0.8, valid_rate=0.1)

        with open(files, 'rb') as handle:
            meta = pickle.load(handle)

        with open(files1, 'rb') as handle:
            meta_k = pickle.load(handle)
    else:
        with open(files, 'rb') as handle:
            meta = pickle.load(handle)

        with open(files1, 'rb') as handle:
            meta_k = pickle.load(handle)

    meta_s =np.concatenate((meta, meta_k), axis=1)

    train_model(opt.data_name, meta_s)



