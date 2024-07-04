import torch
import warnings
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, num_layers=4, dim_h=64):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.dim_h = dim_h

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim=dim_h)
        self.item_embedding = nn.Embedding(num_embeddings=self.num_items, embedding_dim=dim_h)
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, edge_index):
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [emb]

        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        emb_final = 1/(self.num_layers+1) * torch.mean(torch.stack(embs, dim=1), dim=1)

        emb_users_final, emb_items_final = torch.split(emb_final, [self.num_users, self.num_items])

        return emb_users_final, emb_items_final

    def get_embeddings(self, user_indices=None, item_indices=None):
        """
        Retrieve embeddings for specified users or items.

        Args:
            user_indices (Tensor, optional): Indices of users to retrieve embeddings for.
            item_indices (Tensor, optional): Indices of items to retrieve embeddings for.

        Returns:
            Tensor: Embeddings for the specified users or items.
        """

        if (user_indices is not None and item_indices is not None):
            return self.user_embedding.weight[user_indices], self.item_embedding.weight[item_indices]
        
        elif user_indices is not None:
            return self.user_embedding.weight[user_indices], None
        
        elif item_indices is not None:
            return None, self.item_embedding.weight[item_indices]
        
        warnings.warn("Both indices not provided", UserWarning)
        return None, None      
    
if __name__ == "__main__":
    # Number of users and items
    num_users = 10
    num_items = 15

    # Create a random edge index for a simple graph
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                               [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)

    # Initialize the LightGCN model
    model = LightGCN(num_users, num_items)

    # Forward pass
    user_embedding, item_embedding = model.forward(edge_index)
    print("User Embeddings:\n", user_embedding)
    print("Item Embeddings:\n", item_embedding)

    # Get embeddings for specific users and items
    user_indices = torch.tensor([0, 1, 2])
    item_indices = torch.tensor([3, 4, 5, 6])
    specific_user_embedding, specific_item_embedding = model.get_embeddings(user_indices=user_indices, item_indices=item_indices)

    print("Specific User Embeddings:\n", specific_user_embedding)
    print("Specific Item Embeddings:\n", specific_item_embedding)