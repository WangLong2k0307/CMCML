import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from abc import abstractmethod
from utils import load_category, get_category_by_iid


class CML(nn.Module):

    def __init__(self,
                 args, 
                 num_users, 
                 num_items,
                 num_cates,
                 margin):
        super(CML, self).__init__()

        self.num_users = num_users
        self.num_items = num_items 
        self.num_cates = num_cates        
        assert args.per_user_k != 0, 'per_user_k should be greater than zero!'

        self.per_user_embed_k = args.per_user_k
        
        # user embeddings
        self.user_embeddings = nn.Embedding(num_users, self.per_user_embed_k * args.dim)
        nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        
        self.item_embeddings = nn.Embedding(num_items, args.dim, max_norm=args.max_norm)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std= 1.0 / (args.dim ** 0.5))
        self.cate_embeddings = nn.Embedding(num_cates, 10)
        nn.init.normal_(self.cate_embeddings.weight, mean=0, std=0.01)
        gate_vectors = torch.empty(self.per_user_embed_k, 10).cuda()
        torch.nn.init.orthogonal_(gate_vectors, gain=1)
        self.register_parameter("gate_vectors", torch.nn.Parameter(gate_vectors))
        self.data = load_category(args, data_name = 'category.dat')
        self.items = torch.LongTensor(list(range(self.num_items))).cuda()
        self.temperature = args.temperature
        
        self.margin = margin
        
        self.max_norm = args.max_norm
        self.reg = args.reg
        self.reg2 = args.reg2
        self.dim = args.dim 
    def ClipUserNorm(self):
        with torch.no_grad():
            user_embeddings_weight = self.user_embeddings.weight.data
            user_embeddings_weight = user_embeddings_weight.view(self.num_users, 
                                                                 self.per_user_embed_k, 
                                                                 self.dim)

            user_embeddings_weight *= self.max_norm / torch.norm(user_embeddings_weight, 
                                                                p=2, 
                                                                dim=-1,
                                                                keepdim=True)

            self.user_embeddings.weight.data = user_embeddings_weight.view(self.num_users, -1)
            
            
            gate_vectors = self.gate_vectors
            gate_vectors *= self.max_norm / torch.norm(gate_vectors, 
                                                       p=2, dim=-1, keepdim=True)
            self.gate_vectors = gate_vectors

            cate_embeddings_weight = self.cate_embeddings.weight.data
            cate_embeddings_weight *= self.max_norm / torch.norm(cate_embeddings_weight, 
                                                                p=2, 
                                                                dim=-1,
                                                                keepdim=True)
            self.cate_embeddings.weight.data = cate_embeddings_weight

    def preference_loss(self, user_ids, pos_ids, neg_ids):
        pass

    def forward(self, user_ids, pos_ids, neg_ids):
        
        loss, loss_difference, user_loss_diff = self.preference_loss(user_ids, pos_ids, neg_ids)

        loss += self.reg * loss_difference
        loss += self.reg2 * user_loss_diff
        return loss

    
    def predict(self, user_ids):
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids).cuda()
        user_embeddings = self.user_embeddings(user_ids).cuda()
        user_embeddings = user_embeddings.view(user_ids.shape[0], self.per_user_embed_k, self.dim).unsqueeze(-2) 
        item_embeddings = self.item_embeddings.weight # (N, dim)
        item_embeddings = item_embeddings.cuda()
        item_embeddings = item_embeddings.view(1, 1, self.num_items, self.dim) # (1, 1, N, dim)
        scores = torch.square(user_embeddings - item_embeddings).sum(-1) # (batch, k, N)
        gate1, _, _ = self.get_gate(self.data, self.items)
        gate = gate1.t()
        gate = gate.repeat(len(user_ids), 1, 1)
        scores = torch.sum(gate * scores, 1)
        return -scores

    def get_gate(self, data, iids):
        cate_ids = get_category_by_iid(data, iids.cpu().tolist())
        cate_ids = torch.LongTensor(cate_ids).cuda()

        cate_embeddings = self.cate_embeddings(cate_ids)
        gate_output = cate_embeddings.matmul(self.gate_vectors.transpose(0, 1))

        gate_output = self.temp_softmax(gate_output, self.temperature)
        return gate_output, cate_ids, cate_embeddings
    def temp_softmax(self, logits, temperature):
        exp_logits = torch.exp(logits / temperature)
        return exp_logits / exp_logits.sum(dim=-1, keepdim=True)


# CMCML in the paper
class CMCML(CML):

    def __init__(self,
                 args, 
                 num_users, 
                 num_items,
                 num_cates,
                 margin):
        
        assert args.sampling_strategy == 'hard', 'this class is used to hard negative sampling strategy!'
        assert args.per_user_k != 0, 'per_user_k should be greater than zero!'

        super(CMCML, self).__init__(args, num_users, num_items, num_cates, margin)

    
    def preference_loss(self, user_ids, pos_ids, neg_ids):
        
        batch_size = user_ids.shape[0]
        n_negatives = neg_ids.shape[1]
        neg_ids1 = neg_ids.t().contiguous().view(-1) # 
        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, k * dim)
        pos_embeddings1 = self.item_embeddings(pos_ids).cuda() # (batch, dim) 
        user_embeddings = user_embeddings.view(batch_size, self.per_user_embed_k, self.dim) # (batch, k, dim)
        pos_embeddings = pos_embeddings1.unsqueeze(1).expand_as(user_embeddings) # (batch, k, dim)
        pos_distances = torch.square(user_embeddings - pos_embeddings).sum(-1) # (batch, k)
        neg_embeddings1 = self.item_embeddings(neg_ids1).cuda() # (batch * n_negatives, dim)
        neg_embeddings = neg_embeddings1.unsqueeze(1) # (batch*n_negatives, 1, dim)
        pos_embeddings2 = pos_embeddings1.repeat(n_negatives, 1)
        user_embeddings_with_neg = user_embeddings.repeat(n_negatives, 1, 1)
        neg_distances = torch.square(user_embeddings_with_neg - neg_embeddings).sum(-1) 
        neg_distances = neg_distances.view(n_negatives, batch_size, self.per_user_embed_k) # (n_negatives, batch, k)

        pos_gate, cate_ids, pos_cate_embedding = self.get_gate(self.data, pos_ids) # (batch, k)
        best_pos_distances = torch.sum(pos_gate * pos_distances, 1) # batch
        _, k = torch.min(pos_distances, dim=1)
        neg_gates, _, neg_cate_embeddings = self.get_gate(self.data, neg_ids1) # (batch * negatives, k)
        neg_gates = neg_gates.view(n_negatives, batch_size, self.per_user_embed_k)
        final_neg_distances = torch.sum(neg_gates * neg_distances, -1) # (n_negatives, batch)
        min_neg_distances, s = torch.min(final_neg_distances, dim=0) # (batch, )
        embedding_loss = self.margin + best_pos_distances - min_neg_distances # (batch, )
        loss = nn.ReLU()(embedding_loss).mean()

        pos_difference = torch.square(pos_embeddings1.unsqueeze(0) - pos_embeddings1.unsqueeze(1)).sum(-1)
        mask_same = torch.eq(cate_ids.unsqueeze(0), cate_ids.unsqueeze(1))
        mask_diff = 1.0 - mask_same.float()
        loss_difference = torch.log(torch.exp(-2 * F.relu(pos_difference * mask_diff.cuda())).mean())

        interest_difference = torch.square(user_embeddings.unsqueeze(1) - user_embeddings.unsqueeze(2)).sum(-1)
        interest_mask = 1.0 - torch.eye(self.per_user_embed_k)
        interest_mask = interest_mask.unsqueeze(0).expand(batch_size, -1, -1)
        user_loss_diff = torch.log(torch.exp(-2 * F.relu(interest_difference * interest_mask.cuda())).mean())

        return loss, loss_difference, user_loss_diff
    