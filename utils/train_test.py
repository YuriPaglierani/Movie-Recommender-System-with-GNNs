import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch import nn, optim, Tensor
# Unfortunately, for the current version of torch geometric we cannot use the structured_negative_sampling function for Bipartite Graphs:(
# from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from tqdm import tqdm
from sklearn.metrics import ndcg_score

def bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items, bpr_coeff=1e-6):
    reg_loss = bpr_coeff * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))

    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

    bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    
    return -bpr_loss + reg_loss

def sample_negative(pos_idx, num_items, device):
    # this is a fake negative sampling function, but for sparse graphs this can work

    neg_item_index = torch.randint(0, num_items, 
                                  (pos_idx.shape[0], ), device=device) 
     
    return neg_item_index

def train(model, optimizer, train_loader, val_data, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        final_user_emb, final_item_emb = model(batch.edge_index)
        initial_user_emb, initial_item_emb = model.user_embedding.weight, model.item_embedding.weight
    
        user_indices, pos_item_indices, neg_item_indices = get_triplets_indices(batch.edge_index, 
                                                                                model.num_users, 
                                                                                model.num_items)

        print("in the train loop")
        
        # User Embeddings
        final_user_emb, initial_user_emb = final_user_emb[user_indices], initial_user_emb[user_indices]
        # Positive Sampling
        final_pos_item_emb, initial_pos_item_emb = final_item_emb[pos_item_indices], initial_item_emb[pos_item_indices]
        # Negative Sampling
        final_neg_item_emb, initial_neg_item_emb = final_item_emb[neg_item_indices], initial_item_emb[neg_item_indices]

        train_loss = bpr_loss(final_user_emb, initial_user_emb, 
                              final_pos_item_emb, initial_pos_item_emb, 
                              final_neg_item_emb, initial_neg_item_emb)

        train_loss.backward()
        optimizer.step()
            
        total_loss += train_loss.item()
        break

    # in the val_loss we will use the last train batch for deleting some edges
    val_loss, val_ndcg, val_recall = evaluate(model, batch, val_data, device)
    
    
    return total_loss / len(train_loader), val_loss, val_ndcg, val_recall

def get_triplets_indices(edge_index, num_users, num_items):

    user_indices = edge_index[0, edge_index[0] < num_users]
    pos_item_indices = edge_index[1, edge_index[1] >= num_users] - num_users
    neg_item_indices = sample_negative(pos_item_indices, num_items, device)
        
    return user_indices, pos_item_indices, neg_item_indices

def filter_common_edges(A, B):

    # Sort each edge's nodes
    A_sorted, _ = torch.sort(A, dim=0)
    B_sorted, _ = torch.sort(B, dim=0)
    
    # Compute a unique hash for each edge
    max_node = torch.max(torch.max(A), torch.max(B))
    A_hash = A_sorted[0] * (max_node + 1) + A_sorted[1]
    B_hash = B_sorted[0] * (max_node + 1) + B_sorted[1]
    
    # Use torch.isin to find common edges
    mask = torch.isin(A_hash, B_hash)
    print(f"11111 {A.shape}")
    # Step 4: Apply the mask to the original A, avoiding B's common edges
    A_filtered = A[:, ~mask]
    print(f"22222 {A_filtered.shape}")
    return A_filtered
    
def evaluate(model, train_data, test_data, device, k=20):
    model.eval()
    
    all_ndcg = []
    all_recall = []
    test_loss = 0
    with torch.no_grad():
        train_data.to(device)
        test_data = test_data.to(device)

        final_user_emb, final_item_emb = model(test_data.edge_index)
        initial_user_emb, initial_item_emb = model.user_embedding.weight, model.item_embedding.weight
        
        user_indices, pos_item_indices, neg_item_indices = get_triplets_indices(test_data.edge_index, 
                                                                                model.num_users, 
                                                                                model.num_items)
        
        # User Embeddings
        final_user_emb, initial_user_emb = final_user_emb[user_indices], initial_user_emb[user_indices]
        # Positive Sampling
        final_pos_item_emb, initial_pos_item_emb = final_item_emb[pos_item_indices], initial_item_emb[pos_item_indices]
        # Negative Sampling
        final_neg_item_emb, initial_neg_item_emb = final_item_emb[neg_item_indices], initial_item_emb[neg_item_indices]

        test_loss += bpr_loss(final_user_emb, initial_user_emb, 
                                final_pos_item_emb, initial_pos_item_emb, 
                                final_neg_item_emb, initial_neg_item_emb).item()

        # take the unique users, and items in the test set
        unique_users = user_indices.unique()

        # we will take only 10% of the users, for saving time 
        l_users = unique_users.shape[0] // 1000
        l_items = pos_item_indices.unique().shape[0] // 10

        bench_users = torch.randperm(unique_users.shape[0])[:l_users].sort().values
        bench_items = torch.randperm(pos_item_indices.unique().shape[0])[:l_items].sort().values

        # get all possible combinations of users and items
        score_indexes = torch.cartesian_prod(bench_users, bench_items)
        
        mask = (train_data.edge_index[0] < model.num_users) & (train_data.edge_index[1] >= model.num_users)
        seen_edges = train_data.edge_index[:, mask]
        seen_edges[1] = seen_edges[1] - model.num_users
        print(f"seen {seen_edges.shape}")
        score_indexes = filter_common_edges(score_indexes, seen_edges)

        print(f"score_indexes: {score_indexes}")
        scores = torch.mul(final_user_emb[score_indexes[0]], final_item_emb[score_indexes[1]]).sum(dim=-1)
        print(scores.shape)
        # Get top-k recommendations
        _, top_k_indices = torch.topk(scores, k=k)
        
        # Compute NDCG and Recall
        for i, user in enumerate(bench_users):
            mask = (score_indexes[0] == user)
            user_interactions = score_indexes[1][mask]
            recommended_items = top_k_indices[i]
            
            relevance = torch.isin(recommended_items, user_interactions).float()
            ideal_relevance = torch.ones_like(relevance)
            
            ndcg = ndcg_score([relevance.cpu().numpy()], [ideal_relevance.cpu().numpy()])
            recall = (relevance.sum() / len(user_interactions)).item()
            
            all_ndcg.append(ndcg)
            all_recall.append(recall)
    
    return test_loss, np.mean(all_ndcg), np.mean(all_recall)

def train_model(model, train_loader, val_data, test_data, device, epochs=1, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_ndcg = 0
    for epoch in range(epochs):
        loss, val_loss, val_ndcg, val_recall = train(model, optimizer, train_loader, val_data, device)

        scheduler.step(val_ndcg)

        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, \
              Val Loss: {val_loss:.4f}, Val NDCG@10: {val_ndcg:.4f}, Val Recall@10: {val_recall:.4f}')

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_ndcg, test_recall = evaluate(model, test_data, device)
    # print(f'Test NDCG@20: {test_ndcg:.4f}, Test Recall@20: {test_recall:.4f}')
    print(f'Test Loss: {test_loss:.4f}, Test NDCG@10: {test_ndcg:.4f}, Test Recall@10: {test_recall:.4f}')
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
    for batch in train_loader:
        print(batch.edge_index[1].max(), batch.edge_index[1].min())
        break
    num_users, num_items = data_handler.get_num_users_items()
    model = LightGCN(num_users, num_items, dim_h=64, num_layers=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    trained_model = train_model(model, train_loader, val_data, test_data, device)