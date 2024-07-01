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

        self.reset_final_embeddings()
        
    def reset_final_embeddings(self):
        self.final_embedding_counter = 0
        self.final_user_embedding = None
        self.final_item_embedding = None
        
    def forward(self, edge_index):
        emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [emb]

        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        emb_final = 1/(self.num_layers+1) * torch.mean(torch.stack(embs, dim=1), dim=1)

        emb_users_final, emb_items_final = torch.split(emb_final, [self.num_users, self.num_items])

        return emb_users_final, emb_items_final

    def build_final_embeddings(self, edge_index):
        """
        Build final user and item embeddings.

        Args:
            edge_index (Tensor): Edge index of the graph.

        Returns:
            None
        """
        final_user_embedding, final_item_embedding = self.forward(edge_index)
        if self.final_embedding_counter == 0:
            self.final_user_embedding, self.final_item_embedding = final_user_embedding, final_item_embedding
        else:
            self.final_user_embedding += 1/(self.final_embedding_counter+1) * (final_user_embedding - self.final_user_embedding)
            self.final_item_embedding += 1/(self.final_embedding_counter+1) * (final_item_embedding - self.final_item_embedding)

        self.final_embedding_counter += 1

    def get_initial_embeddings(self, user_indices=None, item_indices=None):
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

    def get_final_embeddings(self, user_indices=None, item_indices=None):
        """
        Retrieve final embeddings for specified users or items.

        Args:
            user_indices (Tensor, optional): Indices of users to retrieve embeddings for.
            item_indices (Tensor, optional): Indices of items to retrieve embeddings for.

        Returns:
            Tensor: Final embeddings for the specified users or items.
        """
        if self.final_embedding_counter == 0:
            warnings.warn("Final embeddings not built", UserWarning)
            return None, None

        if (user_indices is not None and item_indices is not None):
            return self.final_user_embedding[user_indices], self.final_item_embedding[item_indices]
        
        elif user_indices is not None:
            return self.final_user_embedding[user_indices], None
        
        elif item_indices is not None:
            return None, self.final_item_embedding[item_indices]
        
        warnings.warn("Both indices not provided", UserWarning)
        return None, None        

    def predict(self, user_indices, item_indices):
        """
        Predict the interaction scores for given user and item indices.

        Args:
            user_indices (Tensor): Indices of users.
            item_indices (Tensor): Indices of items.

        Returns:
            Tensor: Predicted interaction scores for the given user and item indices.
        """
        if self.final_embedding_counter > 0:
            with torch.no_grad():
                user_embedding, item_embedding = self.get_final_embeddings(user_indices, item_indices)
                return torch.matmul(user_embedding, item_embedding.t())
            
        raise ValueError("Final embeddings not built yet. Call build_final_embeddings() first.")
    
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
    specific_user_embedding, specific_item_embedding = model.get_initial_embeddings(user_indices=user_indices, item_indices=item_indices)

    print("Specific User Embeddings:\n", specific_user_embedding)
    print("Specific Item Embeddings:\n", specific_item_embedding)

    # Build final embeddings
    model.build_final_embeddings(edge_index)

    # Predict interaction scores
    interaction_scores = model.predict(user_indices, item_indices)
    print(f"Users: {user_indices.numpy()} Items: {item_indices.numpy()}")
    print("Interaction Scores:\n", interaction_scores)