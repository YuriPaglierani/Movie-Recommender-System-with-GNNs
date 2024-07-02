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

def get_user_items(edge_index):
    user_items = dict()
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_items:
            user_items[user] = []
        user_items[user].append(item)
    return user_items

def compute_recall_at_k(items_ground_truth, items_predicted):
    num_correct_pred = np.sum(items_predicted, axis=1)
    num_total_pred = np.array([len(items_ground_truth[i]) for i in range(len(items_ground_truth))])

    recall = np.mean(num_correct_pred / num_total_pred)

    return recall

def compute_ndcg_at_k(items_ground_truth, items_predicted, K=20):
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

# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices):

    ratings = torch.matmul(model.emb_users.weight, model.emb_items.weight.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        ratings[exclude_users, exclude_items] = -1024

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(ratings, k=K)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    items_predicted = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        items_predicted.append(label)

    recall = compute_recall_at_k(test_user_pos_items_list, items_predicted)
    ndcg = compute_ndcg_at_k(test_user_pos_items_list, items_predicted)

    return recall, ndcg

def bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items, bpr_coeff=1e-6):
    reg_loss = bpr_coeff * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))

    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

    bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    
    return -bpr_loss + reg_loss

def sample_mini_batch(edge_index, batch_size=1024):
    # Generate BATCH_SIZE random indices
    index = np.random.choice(range(edge_index.shape[1]), size=batch_size)

    # Generate negative sample indices
    edge_index = structured_negative_sampling(edge_index)
    edge_index = torch.stack(edge_index, dim=0)
    
    user_index = edge_index[0, index]
    pos_item_index = edge_index[1, index]
    neg_item_index = edge_index[2, index]
    
    return user_index, pos_item_index, neg_item_index

# wrapper function to evaluate model
def test(model, edge_index, exclude_edge_indices):

    emb_users_final, emb_users, emb_items_final, emb_items = model.forward(edge_index)
    user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(edge_index, contains_neg_self_loops=False)

    emb_users_final, emb_users = emb_users_final[user_indices], emb_users[user_indices]

    emb_pos_items_final, emb_pos_items = emb_items_final[pos_item_indices], emb_items[pos_item_indices]
    emb_neg_items_final, emb_neg_items = emb_items_final[neg_item_indices], emb_items[neg_item_indices]

    loss = bpr_loss(emb_users_final, emb_users, emb_pos_items_final, emb_pos_items, emb_neg_items_final, emb_neg_items).item()

    recall, ndcg = get_metrics(model, edge_index, exclude_edge_indices)

    return loss, recall, ndcg

def train(model, optimizer, train_loader, device, batch_size=1024):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        nodes = torch.unique(batch.edge_index)
        final_user_emb, final_item_emb = model(batch.edge_index)
        initial_user_emb, initial_item_emb = model.user_embedding.weight, model.item_embedding.weight
            
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(cluster.edge_index, batch_size)

        final_user_emb, initial_user_emb = final_user_emb[user_indices], initial_user_emb[user_indices]
        # Positive Sampling
            final_pos_item_emb, initial_pos_item_emb = final_item_emb[pos_item_indices], initial_item_emb[pos_item_indices]
            # Negative Sampling
            final_neg_item_emb, initial_neg_item_emb = final_item_emb[neg_item_indices], initial_item_emb[neg_item_indices]

            train_loss = bpr_loss(final_user_emb, initial_user_emb, final_pos_item_emb, initial_pos_item_emb, final_neg_item_emb, initial_neg_item_emb)

            train_loss.backward()
            optimizer.step()
            
            total_loss += train_loss.item()
        
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

def train_model(model, train_loader, val_loader, test_loader, device, epochs=1, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
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