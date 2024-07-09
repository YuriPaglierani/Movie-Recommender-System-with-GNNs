import torch
import torch.optim as optim
import numpy as np
# Unfortunately, for the current version of torch geometric we cannot use the structured_negative_sampling function for Bipartite Graphs:(
# from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from helpers import get_triplets_indices
from typing import Tuple

# for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def bpr_loss(emb_users_final: torch.Tensor, emb_users: torch.Tensor, 
             emb_pos_items_final: torch.Tensor, emb_pos_items: torch.Tensor, 
             emb_neg_items_final: torch.Tensor, emb_neg_items: torch.Tensor, 
             bpr_coeff: float = 5e-3) -> torch.Tensor:
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

    reg_loss = bpr_coeff * (emb_users * emb_users +
                        emb_pos_items * emb_pos_items +
                        emb_neg_items * emb_neg_items).mean()

    normalized_users = normalize_embedding(emb_users_final)
    normalized_pos_items = normalize_embedding(emb_pos_items_final)
    normalized_neg_items = normalize_embedding(emb_neg_items_final)

    cosine_similarity_pos = torch.sum(normalized_users * normalized_pos_items, dim=1)
    cosine_similarity_neg = torch.sum(normalized_users * normalized_neg_items, dim=1)

    bpr_loss = torch.mean(torch.nn.functional.softplus(10*(cosine_similarity_pos - cosine_similarity_neg)))/10.

    return -bpr_loss + reg_loss

def normalize_embedding(emb: torch.Tensor):
    """
    Normalize the embedding.

    Args:
        emb (torch.Tensor): Input embedding.

    Returns:
        torch.Tensor: Normalized embedding.
    """

    return emb / torch.norm(emb, p=2, dim=1, keepdim=True)
    
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
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

def evaluate(model: torch.nn.Module, test_data: torch.Tensor, device: torch.device, top_k: int=100) -> float:
    """
    Evaluate the model on a test set.

    Args:
        model (torch.nn.Module): The model used.
        test_data (torch.Tensor): Tensor containing the test data.
        device (torch.device): Device to use for computation.
        top_k (int): integer for the topk recall metric

    Returns:
        float: Training loss
    """

    model.eval()
    
    with torch.no_grad():
        test_data = test_data.to(device)

        embs = compute_embeddings(model, test_data, device)
        test_loss = bpr_loss(*embs).item()
        user_embs = embs[1]
        item_pos_embs = embs[3]
        item_neg_embs = embs[5]
        embs = (user_embs, item_pos_embs, item_neg_embs)
        recall_at_k = compute_recall_at_k(embs, k=top_k)

    return test_loss, recall_at_k

def compute_recall_at_k(embs, k: int = 20, num_samples: int = 10, sample_size: int = 100) -> float:
    """
    Compute Recall@k given embeddings by sampling users in batches.

    Args:
        embs (tuple): A tuple containing user embeddings and item embeddings.
        k (int): The number of top items to consider for recall calculation.
        num_samples (int): The number of user samples to draw.
        sample_size (int): The number of users to sample in each draw.

    Returns:
        float: Recall@k
    """
    user_embs, pos_item_embs, neg_item_embs = embs

    pos_item_embs_norm = normalize_embedding(pos_item_embs)
    neg_item_embs_norm = normalize_embedding(neg_item_embs)

    num_users = user_embs.size(0)
    total_recall = 0.0

    for _ in range(num_samples):
        sampled_indices = np.random.choice(num_users, sample_size, replace=False)
        user_embs_sampled = user_embs[sampled_indices]
        user_normalized = normalize_embedding(user_embs_sampled)

        user_item_scores_sampled = torch.mm(user_normalized, torch.cat((pos_item_embs_norm, neg_item_embs_norm)).t())

        pos_mask_sampled = torch.zeros_like(user_item_scores_sampled)
        pos_mask_sampled[:, :pos_item_embs.size(0)] = 1 

        # Get top-k scores and their indices for the sampled users
        _, top_k_indices_sampled = torch.topk(user_item_scores_sampled, k, dim=1)

        # Extract the relevant labels for the sampled users
        top_k_labels_sampled = torch.gather(pos_mask_sampled, 1, top_k_indices_sampled)

        # Compute recall for the sampled users
        num_relevant_items_sampled = pos_item_embs.size(0)
        num_relevant_retrieved_sampled = top_k_labels_sampled.sum(dim=1)

        recall_sampled = num_relevant_retrieved_sampled / num_relevant_items_sampled
        total_recall += recall_sampled.mean().item()

    # Compute the mean recall over all samples
    recall_at_k = total_recall / num_samples

    return recall_at_k

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
    hist_train_loss = []
    hist_val_loss = []
    hist_val_recall = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_recall = 0

    for epoch in tqdm(range(epochs)):
        loss = train(model, optimizer, train_loader, device)
        val_loss, recall_at_k = evaluate(model, val_data, device)

        hist_train_loss.append(loss)
        hist_val_loss.append(val_loss)
        hist_val_recall.append(recall_at_k)

        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, \
              Val Loss: {val_loss:.4f}, Recall@k: {recall_at_k:.6f}, k=100')
        if recall_at_k > best_recall:
            best_recall = recall_at_k
            torch.save(model.state_dict(), 'best_model.pth')

    test_loss, recall_at_k = evaluate(model, test_data, device)
    print(f'Test Loss: {test_loss:.4f}, Recall@k: {recall_at_k:.6f}, k=100')

    return model, hist_train_loss, hist_val_loss, hist_val_recall

# Usage example
if __name__ == "__main__":
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)
    
    from data.dataset_handler import MovieLensDataHandler
    from models.light_gcn import LightGCN
    from visualizations import plot_histories

    data_handler = MovieLensDataHandler('data/movielens-25m/ratings.csv', 'data/movielens-25m/movies.csv')
    train_loader, val_data, test_data = data_handler.get_data_training()
    num_users, num_items = data_handler.get_num_users_items()
    model = LightGCN(num_users, num_items, dim_h=64, num_layers=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # if there are parameters load them
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    trained_model, hist_train_loss, hist_val_loss, hist_val_recall = train_model(model, train_loader, val_data, test_data, device, epochs=3)

    np.save('data/histories/hist_train_loss.npy', hist_train_loss)
    np.save('data/histories/hist_val_loss.npy', hist_val_loss)
    np.save('data/histories/hist_val_recall.npy', hist_val_recall)

    plot_histories()