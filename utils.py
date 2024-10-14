import numpy as np 
import pandas as pd
import torch 
from collections import defaultdict 
from tqdm import tqdm
import argparse
import os 
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix,lil_matrix

def load_category(args, data_name = 'category.dat'):
    data_path = os.path.join(args.data_path, data_name)
    data = pd.read_csv(data_path, sep='\t', header=None)
    return data[0].tolist()

def load_data(args, mylog, data_name = 'users.dat', threholds=5):
    data_path = os.path.join(args.data_path, data_name)
    user_dict = defaultdict(set)
    num_users, num_items = 0, 0
    for u, u_liked_items in enumerate(open(data_path).readlines()):
        
        items = u_liked_items.strip().split()
        if int(items[0]) < threholds:
            continue

        user_dict.setdefault(num_users, set())

        for item in items[1:]:
            user_dict[num_users].add(int(item))
            num_items = max(num_items, int(item) + 1)
        
        num_users += 1

    mylog.info('number of users: {}'.format(num_users))
    mylog.info('number of items: {}'.format(num_items))
    
    _num_users = 0
    user_item_matrix = dok_matrix((num_users, num_items), dtype=np.int32)
    for u, u_liked_items in enumerate(open(data_path).readlines()):
        items = u_liked_items.strip().split()

        if int(items[0]) < threholds:
            continue
        for item in items[1: ]:
            user_item_matrix[_num_users, int(item)] = 1
        _num_users += 1
    assert num_users == _num_users
    return user_item_matrix, num_users, num_items

def split_train_val_test(user_item_matrix, args, cur_log, threholds = 5):
    np.random.seed(args.random_seed)

    train_matrix = dok_matrix(user_item_matrix.shape)
    val_matrix = dok_matrix(user_item_matrix.shape)
    test_matrix = dok_matrix(user_item_matrix.shape)

    user_item_matrix = lil_matrix(user_item_matrix)
    num_users = user_item_matrix.shape[0]

    for u in tqdm(range(num_users), desc = "Split data into train/valid/test"):
        items = list(user_item_matrix.rows[u])
        if len(items) < threholds: continue
        np.random.shuffle(items)
        train_count = int(len(items) * args.split_ratio[0] / sum(args.split_ratio))
        valid_count = int(len(items) * args.split_ratio[1] / sum(args.split_ratio))
        
        for i in items[0:train_count]:
            train_matrix[u, i] = 1
        for i in items[train_count:train_count + valid_count]:
            val_matrix[u, i] = 1
        for i in items[train_count + valid_count:]:
            test_matrix[u ,i] = 1

    cur_log.info("total interactions: {}".format(len(train_matrix.nonzero()[0]) + len(val_matrix.nonzero()[0]) + len(test_matrix.nonzero()[0])))
    cur_log.info("split the data into trian/validatin/test {}/{}/{} ".format(
        len(train_matrix.nonzero()[0]),
        len(val_matrix.nonzero()[0]),
        len(test_matrix.nonzero()[0])))
    return train_matrix, val_matrix, test_matrix

def get_category_by_iid(data, iid):
    return [data[id] for id in iid]

def get_category_count(args, data_name = 'category.dat'):
    data_path = os.path.join(args.data_path, data_name)
    max_value = 0
    with open(data_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            elements = list(map(int, elements))
            line_max = max(elements)
            max_value = max(max_value, line_max)
    return max_value + 1

def load_dok_matrix(data, num_users, num_items):
    user_item_matrix = dok_matrix((num_users, num_items), dtype=np.int32) 
    with open(data, 'r') as file:
        for line in file:
            user, item = line.strip().split('\t')
            user = int(user)
            item = int(item)
            user_item_matrix[user, item] = 1
    return user_item_matrix

def set_seeds(random_seed = 1234):
    
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser(description='config parameters!')
    parser.add_argument('--data_path', type=str, default='data/ciao', help='data u wanna run')
    parser.add_argument('--model', type=str, default='CMCML', help='which model to run')
    parser.add_argument('--eval_user_nums', type = int, default=10000000, help='eval_user_nums')
    parser.add_argument('--margin', type=float, default=1.0, help='safe margin')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--k', type=list, default= [5], help='embedding dimension')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size number')
    parser.add_argument('--epoch', type = int, default=100, help='epoch number')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max norm 4 embeddings')
    parser.add_argument('--split_ratio', type=tuple, default=(3, 1, 1), help='split_ratio to partition dataset')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer 4 optimization')
    parser.add_argument('--test', action='store_true', help='optimizer 4 optimization')
    parser.add_argument('--num_negs', type=int, default=10, help='number of negative sampling items')
    parser.add_argument('--sampling_strategy', type=str, default='hard', help='sampling strategy')
    parser.add_argument('--per_user_k', type=int, default=5, help='the number of embeddings of users')
    parser.add_argument('--reg', type=float, default=0.1, help='whether regularization')
    parser.add_argument('--reg2', type=float, default=0)
    parser.add_argument('--num_neg_candidates', type=int, default=300)
    parser.add_argument('--spreadout_weight', type=float, default=1.0)
    parser.add_argument('--use_sparse_reg', type=float, default=1., help='l1 sparse')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')

    return parser.parse_args()

__all__ = ['load_data', 'set_seeds', 'split_train_val_test', 'parse_args']


