import torch
import torch.optim as optim
# Unfortunately, for the current version of torch geometric we cannot use the structured_negative_sampling function for Bipartite Graphs:(
# from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from helpers import get_triplets_indices
from typing import Tuple

def bpr_loss(emb_users_final: torch.Tensor, emb_users: torch.Tensor, 
             emb_pos_items_final: torch.Tensor, emb_pos_items: torch.Tensor, 
             emb_neg_items_final: torch.Tensor, emb_neg_items: torch.Tensor, 
             bpr_coeff: float = 1e-6) -> torch.Tensor:
    """
    Compute the Bayesian Personalized Ranking (BPR) loss.

    Args:
        emb_users_final (torch.Tensor): Final user embeddings.
        emb_users (torch.Tensor): Initial user embeddings.
        emb_pos_items_final (torch.Tensor): Final positive item embeddings.
        emb_pos_items (torch.Tensor): Initial positive item embeddings.
        emb_neg_items_final (torch.Tensor): Final negative item embeddings.
        emb_neg_items (torch.Tensor): Initial negative item embeddings.
        bpr_coeff (float): Regularization coefficient.

    Returns:
        torch.Tensor: Computed BPR loss.
    """

    reg_loss = bpr_coeff * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))

    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

    bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    
    return -bpr_loss + reg_loss

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
          train_loader: torch.utils.data.DataLoader, device: torch.device
          ) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): Device to use for computation.

    Returns:
        float: Training loss
    """

    model.train()
    total_loss = 0
    total_w = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        embs = compute_embeddings(model, batch, device)

        train_loss = bpr_loss(*embs)

        train_loss.backward()
        optimizer.step()

        w = batch.edge_index.shape[1]
        total_w += w

        total_loss += train_loss.item() * w

    return total_loss / total_w

def compute_embeddings(model: torch.nn.Module, 
                       data: torch.Tensor, 
                       device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Compute embeddings for users and items.

    Args:
        model (torch.nn.Module): The model to use for computing embeddings.
        data (torch.Tensor): Input data.
        device (torch.device): Device to use for computation.

    Returns:
        Tuple[torch.Tensor, ...]: Tuple containing various embeddings.
    """

    final_user_emb, final_item_emb = model(data.edge_index)
    initial_user_emb, initial_item_emb = model.user_embedding.weight, model.item_embedding.weight
        
    user_indices, pos_item_indices, neg_item_indices = get_triplets_indices(data.edge_index, 
                                                                                model.num_users, 
                                                                                model.num_items, device)
        
    # User Embeddings
    final_user_emb, initial_user_emb = final_user_emb[user_indices], initial_user_emb[user_indices]
    # Positive Sampling
    final_pos_item_emb, initial_pos_item_emb = final_item_emb[pos_item_indices], initial_item_emb[pos_item_indices]
    # Negative Sampling
    final_neg_item_emb, initial_neg_item_emb = final_item_emb[neg_item_indices], initial_item_emb[neg_item_indices]

    return final_user_emb, initial_user_emb, final_pos_item_emb, initial_pos_item_emb, final_neg_item_emb, initial_neg_item_emb

def evaluate(model: torch.nn.Module, test_data: torch.Tensor, device: torch.device) -> float:
    """
    Evaluate the model on a test set.

    Args:
        model (torch.nn.Module): The model used.
        test_data (torch.Tensor): Tensor containing the test data.
        device (torch.device): Device to use for computation.

    Returns:
        float: Training loss
    """

    model.eval()
    
    with torch.no_grad():
        test_data = test_data.to(device)

        embs = compute_embeddings(model, test_data, device)
        test_loss = bpr_loss(*embs).item()

    return test_loss

    
def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                val_data: torch.Tensor, test_data: torch.Tensor, device: torch.device, 
                epochs: int = 1, lr: float = 0.001) -> torch.nn.Module:
    """
    Train the model for multiple epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_data (torch.Tensor): Validation data.
        test_data (torch.Tensor): Test data.
        device (torch.device): Device to use for computation.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.

    Returns:
        torch.nn.Module: Trained model.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        loss = train(model, optimizer, train_loader, device)
        val_loss = evaluate(model, val_data, device)
        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, \
              Val Loss: {val_loss:.4f}')
        
    torch.save(model.state_dict(), 'best_model.pth')
    test_loss = evaluate(model, test_data, device)
    print(f'Test Loss: {test_loss:.4f}')

    return model

# Usage example
if __name__ == "__main__":
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)
    
    from data.dataset_handler import MovieLensDataHandler
    from models.light_gcn import LightGCN

    data_handler = MovieLensDataHandler('data/movielens-25m/ratings.csv', 'data/movielens-25m/movies.csv')
    data_handler.preprocess()
    data_handler.split_data()
    train_loader, val_data, test_data = data_handler.get_data()
    num_users, num_items = data_handler.get_num_users_items()
    model = LightGCN(num_users, num_items, dim_h=64, num_layers=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # if there are parameters load them
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    trained_model = train_model(model, train_loader, val_data, test_data, device, epochs=3)