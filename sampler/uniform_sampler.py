from operator import ne
import numpy as np 
from .sampler import Sampler
from tqdm import tqdm

class UniformSampler(Sampler):
    def _negative_sampling(self, user_item_pairs):

        sampling_triplets = []
        num_items = self.interactions.shape[1]
        for user_id, pos in tqdm(user_item_pairs, desc='===> generate negative samplings...'):
            for _ in range(self.n_negatives):
                neg = np.random.randint(num_items)
                while neg in self.user_items[user_id]:
                    neg = np.random.randint(num_items)
                sampling_triplets.append((user_id, pos, neg))
        
        return sampling_triplets
