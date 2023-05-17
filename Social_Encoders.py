import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, center_user,cluster_list,features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cuda"):
        super(Social_Encoder, self).__init__()

        self.center_user=center_user
        self.cluster_list=cluster_list
        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):

        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])

            for i in self.center_user:
                if int(node) in self.cluster_list[i]:
                    to_neighs.append(self.cluster_list[i])

        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()
        
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)  #按行
        combined = F.relu(self.linear1(combined))

        return combined
