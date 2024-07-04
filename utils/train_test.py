import torch
import torch.optim as optim
import numpy as np
# Unfortunately, for the current version of torch geometric we cannot use the structured_negative_sampling function for Bipartite Graphs:(
# from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from helpers import get_triplets_indices, is_in_feasible, get_user_items
from typing import Tuple, List

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
          train_loader: torch.utils.data.DataLoader, val_data: torch.Tensor, 
          device: torch.device) -> Tuple[float, float, float, float]:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_data (torch.Tensor): Validation data.
        device (torch.device): Device to use for computation.

    Returns:
        tuple[float, float, float, float]: Training loss, validation loss, validation NDCG, validation recall.
    """

    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        embs = compute_embeddings(model, batch, device)

        train_loss = bpr_loss(*embs)

        train_loss.backward()
        optimizer.step()
            
        total_loss += train_loss.item()

    # in the val_loss we will use the last train batch for deleting some edges
    val_loss, val_ndcg, val_recall = evaluate(model, batch, val_data, device)
    
    
    return total_loss / len(train_loader), val_loss, val_ndcg, val_recall

def compute_ndcg_at_k(items_ground_truth: List[List[int]], 
                      items_predicted: List[List[int]], 
                      K: int = 20) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.

    Args:
        items_ground_truth (list[list[int]]): Ground truth items for each user.
        items_predicted (list[list[int]]): Predicted items for each user.
        K (int): Number of items to consider.

    Returns:
        float: Computed NDCG@K score.
    """

    test_matrix = np.zeros((len(items_predicted), K))

    for i, items in enumerate(items_ground_truth):
        length = min(len(items), K)
        test_matrix[i, :length] = 1
    
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, K + 2)), axis=1)
    dcg = items_predicted * (1. / np.log2(np.arange(2, K + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    
    return np.mean(ndcg)

def compute_recall_at_k(items_ground_truth: List[List[int]], items_predicted: List[List[int]]) -> float:
    """
    Compute Recall at K.

    Args:
        items_ground_truth (list[list[int]]): Ground truth items for each user.
        items_predicted (list[list[int]]): Predicted items for each user.

    Returns:
        float: Computed Recall@K score.
    """

    num_correct_pred = np.sum(items_predicted, axis=1)
    num_total_pred = np.array([len(items_ground_truth[i]) for i in range(len(items_ground_truth))])

    recall = np.mean(num_correct_pred / num_total_pred)

    return recall

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
        tuple[torch.Tensor, ...]: Tuple containing various embeddings.
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

def evaluate(model, train_data, test_data, device, k=20):
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        train_data.to(device)
        test_data = test_data.to(device)

        embs = compute_embeddings(model, test_data, device)
        final_user_emb = embs[0]
        final_item_emb = embs[2]
        test_loss += bpr_loss(*embs).item()

        unique_users = test_data.edge_index[0, test_data.edge_index[0] < model.num_users].unique()
        item_indices = test_data.edge_index[1, test_data.edge_index[1] >= model.num_users] - model.num_users

        # we will take only 10% of the users, for saving time 
        l_users = unique_users.shape[0] // 1000
        l_items = item_indices.unique().shape[0] // 10

        bench_users = torch.randperm(unique_users.shape[0])[:l_users].sort().values
        bench_items = torch.randperm(item_indices.unique().shape[0])[:l_items].sort().values

        # get all possible combinations of users and items
        score_indexes = torch.cartesian_prod(bench_users, bench_items).t()
        
        mask = (train_data.edge_index[0] < model.num_users) & (train_data.edge_index[1] >= model.num_users)
        seen_edges = train_data.edge_index[:, mask]
        seen_edges[1] = seen_edges[1] - model.num_users
        
        score_indexes = is_in_feasible(score_indexes, seen_edges)

        scores = torch.mul(final_user_emb[score_indexes[0]], final_item_emb[score_indexes[1]]).sum(dim=-1)

        # Get top-k recommendations
        _, top_k_indices = torch.topk(scores, k=k)
        
        users = score_indexes[0].unique()
        test_user_pos_items = get_user_items(test_data.edge_index)

        ndcgs = []
        recalls = []

        for user in users:
            user_items = test_user_pos_items.get(user.item(), [])
            if not user_items:
                continue
            user_scores = scores[score_indexes[0] == user].cpu().numpy()
            user_items_set = set(user_items)

            # NDCG
            ideal_dcg = np.sum(1 / np.log2(np.arange(2, min(len(user_items), k) + 2)))
            dcg = np.sum([1 / np.log2(i + 2) if item in user_items_set else 0 
                          for i, item in enumerate(user_scores.argsort()[::-1][:k])])
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcgs.append(ndcg)

            # Recall
            top_k_items = set(user_scores.argsort()[::-1][:k])
            recall = len(top_k_items.intersection(user_items_set)) / len(user_items_set)
            recalls.append(recall)

        print(ndcgs)
        ndcg = np.mean(ndcgs) if ndcgs else 0
        recall = np.mean(recalls) if recalls else 0

    # return test_loss, 10, 10
    return test_loss, ndcg, recall

    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_ndcg = 0
    for epoch in tqdm(range(epochs)):
        loss, val_loss, val_ndcg, val_recall = train(model, optimizer, train_loader, val_data, device)

        scheduler.step(val_ndcg)

        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, \
              Val Loss: {val_loss:.4f}, Val NDCG@20: {val_ndcg:.4f}, Val Recall@20: {val_recall:.4f}')

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    # test_loss, test_ndcg, test_recall = evaluate(model, test_data, device)
    # # print(f'Test NDCG@20: {test_ndcg:.4f}, Test Recall@20: {test_recall:.4f}')
    # print(f'Test Loss: {test_loss:.4f}, Test NDCG@20: {test_ndcg:.4f}, Test Recall@20: {test_recall:.4f}')
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
    trained_model = train_model(model, train_loader, val_data, test_data, device, epochs=6)