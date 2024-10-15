import numpy as np 
from collections import defaultdict
import torch as tc 
import torch.nn as nn 
from abc import abstractmethod
from scipy.sparse import lil_matrix
import random
from utils import set_seeds

class Sampler:
    """
    A sampler is responsible for triplet sampling within a specific strategy
    :param name: sampler name
    :param model: current training model
    :param interactions: input user interactions in
           scipy.sparse.lil_matrix format
    :param n_workers: number of workers
    :param n_negatives: number of negatives
    :param batch_size: batch size
    :param kwargs: optional keyword arguments
    """
    @classmethod 
    def _get_popularity(cls, interactions):
        popularity_dict = defaultdict(set)
        for uid, iids in enumerate(interactions.rows):
            for iid in iids:
                popularity_dict[iid].add(uid)

        popularity_dict = {
            key: len(val) for key, val in popularity_dict.items()
        }
        return popularity_dict
    
    def __init__(self, 
                 sampling_strategy, 
                 interactions, 
                 model,
                 n_negatives=10, 
                 random_seed=1234,
                 **kwargs):

        if sampling_strategy not in ['uniform', 'hard']:
            raise ValueError('only support [uniform, hard] now!')

        self.sampling_strategy = sampling_strategy
        self.interactions = lil_matrix(interactions)
        self.n_negatives = n_negatives
        self.random_seed = random_seed
        self.neg_alpha = 1.0
        if kwargs is not None:
            self.__dict__.update(kwargs) 
        self.user_items = {uid: set(iids) for uid, iids in enumerate(
            self.interactions.rows)}
        

    def sampling(self):
        user_positive_item_pairs = np.asarray(self.interactions.nonzero()).T

        return self._negative_sampling(user_positive_item_pairs)
        
    @abstractmethod
    def _candidate_neg_ids(self):
        return np.arange(self.interactions.shape[1])

    @abstractmethod    
    def _negative_sampling(self, user_ids, pos_ids, neg_ids):
        raise NotImplementedError(
            '_negative_sampling method should be implemented in child class')
    

