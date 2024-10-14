import torch 
import numpy as np 
import os 
from Log import MyLog
import gc 
from evaluate import * 
from torch.nn.utils import clip_grad_norm_
from model import CMCML
from utils import *
from utils import load_category, load_dok_matrix, get_category_count
from scipy import sparse
from scipy.sparse import coo_matrix, dok_matrix
from collections import defaultdict
import pandas as pd
torch.cuda.empty_cache()
gc.collect()

SUPPORT_MODEL = {
        'CMCML': CMCML
    }

def load_data(dataset):
    data_path = os.path.join(dataset, 'train.dat')
    data = pd.read_csv(data_path, sep='\t', header=None)
    data_count = data[0].value_counts().to_dict()
    dicted_data_count = dict(sorted(data_count.items()))
    return list(dicted_data_count.values())

def top_user(user_pop, ratio):
    top_user_count = int(len(user_pop) * ratio)
    user_index = {element: position for position, element in enumerate(user_pop)}
    sorted_user_pop = sorted(user_pop)
    top_user_pop = sorted_user_pop[-top_user_count:]
    top_user_index = [user_index[user] for user in top_user_pop]
    return top_user_pop, top_user_index


def test(model, logger, val_users, evaluators, top_rec_ks, epoch=0):
    
    if not isinstance(top_rec_ks, list):
        top_rec_ks = list(top_rec_ks)
    
    all_results = defaultdict(dict)
    
    with torch.no_grad():
        model.eval()
        all_results.setdefault('dataset', {})
        all_results.setdefault('interests', {})
        all_results.setdefault('reg', {})
        all_results.setdefault('reg2', {})
        all_results.setdefault('pre@3', {})
        all_results.setdefault('rec@3', {})
        all_results.setdefault('ndcg@3', {})
        all_results.setdefault('dp@3', {})
        all_results.setdefault('pre@5', {})
        all_results.setdefault('rec@5', {})
        all_results.setdefault('ndcg@5', {})
        all_results.setdefault('dp@5', {})
        all_results.setdefault('pre@10', {})
        all_results.setdefault('rec@10', {})
        all_results.setdefault('ndcg@10', {})
        all_results.setdefault('dp@10', {})
        all_results.setdefault('pre@20', {})
        all_results.setdefault('rec@20', {})
        all_results.setdefault('ndcg@20', {})
        all_results.setdefault('dp@20', {})
        for k in top_rec_ks:
            p_k, r_k, n_k, dp_k = evaluators.max_sum_dispersion(model, val_users, k)
            logger.info("Epoch: {}, precision@{}: {}, recall@{}: {}, ndcg@{}: {}, dispersion@{}: {}".format(epoch, 
                                                                                         k, 
                                                                                         p_k, 
                                                                                         k, 
                                                                                         r_k, 
                                                                                         k, 
                                                                                         n_k,
                                                                                         k,
                                                                                         dp_k))

            all_results['pre@' + str(k)][0] = p_k
            all_results['rec@' + str(k)][0] = r_k
            all_results['ndcg@' + str(k)][0] = n_k
            all_results['dp@' + str(k)][0] = dp_k
            
    return all_results

if __name__ == '__main__':

    args = parse_args()
    set_seeds(args.random_seed)
    
    save_path = os.path.join(args.data_path, 
                             args.model,
                             args.sampling_strategy, 
                             'best',
                             'per_k_{}'.format(args.per_user_k), 
                             'margin_{}'.format(args.margin))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    log_path = '_'.join([
        'lr_{}'.format(args.lr),
        'max_norm_{}'.format(args.max_norm),
        'num_negs_{}'.format(args.num_negs),
        'optim_{}'.format(args.optimizer),
        'reg_{}'.format(args.reg),
        'dim_{}'.format(args.dim),
        'k_{}'.format(args.per_user_k),
        'temperature_{}'.format(args.temperature),
        'reg2_{}'.format(args.reg2)
    ]) 

    cur_log = MyLog(os.path.join(save_path, log_path + '.log'), log_path + '.pth')
    cur_log.info(args)
    if os.path.exists(os.path.join(args.data_path, "np_data")):
        print("load saved data....")
        user_train_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_train_matrix.npz'))
        user_val_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_val_matrix.npz'))
        user_test_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_test_matrix.npz'))

        user_train_matrix = dok_matrix(user_train_matrix)
        user_val_matrix = dok_matrix(user_val_matrix)
        user_test_matrix = dok_matrix(user_test_matrix)

        num_users, num_items = user_train_matrix.shape
        cur_log.info("number of users: {}".format(num_users))
        cur_log.info("number of items: {}".format(num_items))

    else:
        os.makedirs(os.path.join(args.data_path, "np_data"))
        user_item_matrix, num_users, num_items = load_data(args, cur_log, data_name='users.dat', threholds=5)
        # user_train_matrix, user_val_matrix, user_test_matrix = split_train_val_test(user_item_matrix, args, cur_log)
        user_train_matrix = load_dok_matrix(os.path.join(args.data_path, 'train.dat'), num_users, num_items)
        user_val_matrix = load_dok_matrix(os.path.join(args.data_path, 'val.dat'), num_users, num_items)
        user_test_matrix = load_dok_matrix(os.path.join(args.data_path, 'test.dat'), num_users, num_items)
        print('saving splited data')
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_train_matrix.npz'), coo_matrix(user_train_matrix))
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_val_matrix.npz'), coo_matrix(user_val_matrix))
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_test_matrix.npz'), coo_matrix(user_test_matrix))


    train_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_train_matrix)
    val_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_val_matrix)
    test_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_test_matrix)
    num_cates = get_category_count(args=args)
    try:
        model = SUPPORT_MODEL[args.model](args, 
                    num_users,
                    num_items,
                    num_cates,
                    args.margin).cuda()
    except KeyError as e:
        raise e('Do not support model {}'.format(args.model))
    
    
    cur_log.info("Evaluate Model...")
    cur_log.info("Loading Saved Model From {}".format(cur_log.best_model_path))
        
    best_model_path = cur_log.best_model_path
    if not os.path.exists(best_model_path):
        raise ValueError('saved model are not exist at %s' % best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    
    model.eval()

    torch.cuda.empty_cache()
    gc.collect()

    val_users = np.asarray([i for i in range(model.num_users)])
    user_pop = load_data(args.data_path)
    ratio_pop = 1.0
    top_user_pop, top_user_index = top_user(user_pop, ratio_pop)
    top_val_users = np.asarray(top_user_index)
    
    all_results = test(model, cur_log, val_users, test_evaluator, [3, 5, 10, 20])
    all_results['dataset'][0] = args.data_path
    all_results['interests'][0] = args.per_user_k
    all_results['reg'][0] = args.reg
    all_results['reg2'][0] = args.reg2
    all_results = pd.DataFrame(all_results)


    all_results.to_csv(args.data_path + '/result.csv', mode='a', index=False, header=False)


    
