import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch import nn, optim, Tensor
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, ndcg_score

def bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items, bpr_coeff=1e-6):
    reg_loss = bpr_coeff * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))

    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

    bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    
    return -bpr_loss + reg_loss

def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        batch = batch.to(device)
        user_emb, item_emb = model(batch.edge_index)
        
        # Positive examples
        pos_scores = (user_emb[batch.edge_index[0]] * item_emb[batch.edge_index[1]]).sum(dim=1)
        
        # Negative sampling
        neg_edge_index = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes, num_neg_samples=batch.edge_index.size(1))
        # Ensure neg_edge_index is within bounds
        neg_edge_index = neg_edge_index.clamp(0, batch.num_nodes - 1)
        neg_scores = (user_emb[neg_edge_index[0]] * item_emb[neg_edge_index[1]]).sum(dim=1)
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader, device, k=10):
    model.eval()
    
    all_ndcg = []
    all_recall = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            user_emb, item_emb = model(batch.edge_index)
            
            # Compute scores for all items
            scores = torch.matmul(user_emb, item_emb.t())
            
            # Get top-k recommendations
            _, top_k_indices = torch.topk(scores, k=k)
            
            # Compute NDCG and Recall
            for i, user in enumerate(batch.edge_index[0].unique()):
                user_interactions = batch.edge_index[1][batch.edge_index[0] == user]
                recommended_items = top_k_indices[i]
                
                relevance = torch.isin(recommended_items, user_interactions).float()
                ideal_relevance = torch.ones_like(relevance)
                
                ndcg = ndcg_score([relevance.cpu().numpy()], [ideal_relevance.cpu().numpy()])
                recall = (relevance.sum() / len(user_interactions)).item()
                
                all_ndcg.append(ndcg)
                all_recall.append(recall)
    
    return np.mean(all_ndcg), np.mean(all_recall)

def train_model(model, train_loader, val_loader, test_loader, device, epochs=1, lr=0.001, weight_decay=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_ndcg = 0
    for epoch in range(epochs):
        loss = train(model, optimizer, train_loader, device)
        val_ndcg, val_recall = evaluate(model, val_loader, device)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val NDCG@10: {val_ndcg:.4f}, Val Recall@10: {val_recall:.4f}')
        
        scheduler.step(val_ndcg)
        
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_ndcg, test_recall = evaluate(model, test_loader, device)
    print(f'Test NDCG@10: {test_ndcg:.4f}, Test Recall@10: {test_recall:.4f}')
    
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
    
    train_loader, val_loader, test_loader = data_handler.get_dataloaders()
    
    num_users, num_items = data_handler.get_num_users_items()
    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    trained_model = train_model(model, train_loader, val_loader, test_loader, device)