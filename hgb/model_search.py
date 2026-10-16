import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

import random


class ShallowNodeTower(nn.Module):
    """Simple linear projection of a node type's raw features into the shared
    embedding space -- used for the non-searched side of a link-prediction pair
    (e.g. Author/User), which runs no meta-path search of its own."""

    def __init__(self, attr_dim, hidden):
        super(ShallowNodeTower, self).__init__()
        self.proj = nn.Linear(attr_dim, hidden)
        nn.init.xavier_uniform_(self.proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.proj.bias)

    def forward(self, attr):
        return self.proj(attr)


class UserEmbeddingTower(nn.Module):
    """Free per-user trainable vectors for the non-searched side of a
    link-prediction pair -- the same per-user capacity metapath2vec (its
    embedding table) and GAT (aggregation over the user's own train edges) get
    implicitly, so cross-model comparisons isolate the ITEM encoder.
    forward() takes user INDICES: pass src_attr = torch.arange(num_src) and
    every ShallowNodeTower call site (tower(src_attr[idx]), tower(src_attr))
    works unchanged. Transductive: an unseen user keeps an untrained row."""

    def __init__(self, num_users, hidden):
        super(UserEmbeddingTower, self).__init__()
        self.emb = nn.Embedding(num_users, hidden)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, idx):
        return self.emb(idx)


class LMSPS_Se(nn.Module):
    def __init__(self, hidden, nclass, feat_keys, label_feat_keys, tgt_key, dropout, 
                 input_drop, device, num_final, residual=False, bns=False, data_size=None, num_sampled=1, task='classification'):
        
        super(LMSPS_Se, self).__init__()

        self.feat_keys = feat_keys
        self.label_feat_keys = label_feat_keys
        self.num_feats = len(feat_keys)
        self.all_meta_path = list(self.feat_keys) + list(self.label_feat_keys)
        self.num_sampled = num_sampled
        self.num_channels = self.num_sampled
        self.num_paths = len(self.all_meta_path)
        self.num_final = num_final
        self.num_res = self.num_paths - self.num_final
        self.tgt_key = tgt_key
        self.residual = residual

        print("number of paths", len(feat_keys), len(label_feat_keys))

        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(
                torch.Tensor(v, hidden).uniform_(-0.5, 0.5))

        if len(label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(
                    torch.Tensor(nclass, hidden).uniform_(-0.5, 0.5))

        self.task = task
        if self.task == 'classification':
            self.lr_output = nn.Sequential(
                nn.Linear(hidden, nclass, bias=False),
                nn.BatchNorm1d(nclass, affine=bns, track_running_stats=bns)
            )

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.alpha = torch.ones(self.num_paths).to(device)
        self.alpha.requires_grad_(True)

        # eval_gumbel=True (default, upstream behaviour): Gumbel noise is drawn
        # on EVERY forward, also under model.eval(), so each node batch of an
        # evaluation pass gets a different path weighting and the same path set
        # scores differently on every repeat of the sampling evaluation.
        # eval_gumbel=False: under model.eval() use the noise-free path strength
        # softmax(alpha / tau) -- one fixed weighting per evaluation pass.
        self.eval_gumbel = True

        if self.residual:
            self.res_fc = nn.Linear(hidden, hidden)

        self.init_params()

    def init_params(self):

        gain = nn.init.calculate_gain("relu")
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        if self.task == 'classification':
            for layer in self.lr_output:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)


    def alphas(self):
        alphas= [self.alpha]
        return alphas


    def epoch_pool(self, eps):
        """Current dynamic search space (paper Eq. 3): the top
        int(num_res * eps) + num_final paths by alpha; eps=0 gives the final pool."""
        return torch.argsort(self.alpha, dim=-1, descending=True)[:int(self.num_res * eps) + self.num_final]

    def epoch_sample(self, eps):

        indices = self.epoch_pool(eps)

        sampled = random.sample(list(indices.cpu().numpy()), self.num_sampled)
        sampled = sorted(sampled)
        print(f"sampled: {sampled}")
        return sampled
    

    def forward(self, epoch_sampled, feats_dict, label_feats_dict, meta_path_sampled, label_meta_path_sampled):


        if isinstance(feats_dict[meta_path_sampled[-1]], torch.Tensor):
            for k, v in feats_dict.items():
                feats_dict[k] = self.input_drop(v @ self.embeding[k])
            
        elif isinstance(feats_dict[meta_path_sampled[-1]], SparseTensor):
            for k, v in feats_dict.items():
                feats_dict[k] = self.input_drop(v @ self.embeding[k[-1]])
            
        else:
            assert 0



        def get_gumbel_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=-1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=-1)
                index = probs.max(-1, keepdim=True)[-1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (
                        (torch.isinf(gumbels).any())
                        or (torch.isinf(probs).any())
                        or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break

            return probs

        for k, v in label_feats_dict.items():

            label_feats_dict[k] = self.input_drop(v @ self.labels_embeding[k])
            
        x = [feats_dict[k] for k in meta_path_sampled] + [label_feats_dict[k] for k in label_meta_path_sampled]
        x = torch.stack(x, dim=1) # [B, C, D]

        ws = torch.stack([self.alpha[idx] for idx in epoch_sampled])

        if self.training or self.eval_gumbel:
            ws = get_gumbel_prob(ws)
        else:
            # noise-free path strength (Eq. 2 without u_k), renormalised over
            # the sampled set like q'_k in Eq. 4; identical for every batch
            ws = nn.functional.softmax(ws / self.tau, dim=-1)
        x = torch.einsum('bcd,c->bd', x, ws)

        if self.residual:
            k = self.tgt_key
            tgt_feat = feats_dict[k]
            x = x + self.res_fc(tgt_feat)

        x = self.dropout(self.prelu(x))
        if self.task == 'classification':
            x = self.lr_output(x)

        return x

    def set_tau(self, tau):
        self.tau = tau

    def sample(self, keys, label_keys, lam, topn, all_path=False):
        '''
        to sample one candidate edge type per link
        '''
        length = len(self.alpha)
        seq_softmax = None if self.alpha is None else F.softmax(self.alpha, dim=-1)
        max = torch.max(seq_softmax, dim=0).values
        min = torch.min(seq_softmax, dim=0).values
        threshold = lam * max + (1 - lam) * min


        _, idxl = torch.sort(seq_softmax, descending=True)  # descending=False -> ascending, descending=True -> descending

        idx = idxl[:self.num_sampled]

        if all_path:
            path = []
            label_path = []
            for i, index in enumerate(idxl):
                if index < len(keys):
                    path.append((keys[index], i))
                else:
                    label_path.append((label_keys[index - len(keys)], i))
            return [path, label_path], idx

        if topn:
            id_paths = idxl[:topn]
        else:
            id_paths = [k for k in range(length) if seq_softmax[k].item() >= threshold]
        path = [keys[i] for i in range(len(keys)) if i in id_paths]
        label_path = [label_keys[i] for i in range(len(label_keys)) if i+len(keys) in id_paths]

        return [path, label_path], idx





