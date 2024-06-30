import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        self.convs = nn.ModuleList([LGConv(normalize=False) for _ in range(num_layers)])
        
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        x = self.embedding.weight
        
        layer_wise_embedding = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_wise_embedding.append(x)
        
        final_embedding = sum(layer_wise_embedding) / (self.num_layers + 1)
        
        user_embedding, item_embedding = torch.split(final_embedding, [self.num_users, self.num_items])
        
        return user_embedding, item_embedding

    def get_embedding(self, user_indices=None, item_indices=None):
        user_embedding, item_embedding = self.forward(None)
        
        if user_indices is not None:
            return user_embedding[user_indices]
        elif item_indices is not None:
            return item_embedding[item_indices]
        else:
            return user_embedding, item_embedding

    def predict(self, user_indices, item_indices):
        user_embedding, item_embedding = self.get_embedding()
        return (user_embedding[user_indices] * item_embedding[item_indices]).sum(dim=1)