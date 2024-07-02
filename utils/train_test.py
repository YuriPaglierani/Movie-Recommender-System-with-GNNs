import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch import nn, optim, Tensor
# Unfortunately, for the moment we cannot use the structured_negative_sampling function from torch_geometric.utils in Bipartite Graphs :(
# from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from tqdm import tqdm
from sklearn.metrics import ndcg_score

# def get_user_items(edge_index):
#     user_items = dict()
#     for i in range(edge_index.shape[1]):
#         user = edge_index[0][i].item()
#         item = edge_index[1][i].item()
#         if user not in user_items:
#             user_items[user] = []
#         user_items[user].append(item)
#     return user_items

# def compute_recall_at_k(items_ground_truth, items_predicted):
#     num_correct_pred = np.sum(items_predicted, axis=1)
#     num_total_pred = np.array([len(items_ground_truth[i]) for i in range(len(items_ground_truth))])

#     recall = np.mean(num_correct_pred / num_total_pred)

#     return recall

# def compute_ndcg_at_k(items_ground_truth, items_predicted, K=20):
#     test_matrix = np.zeros((len(items_predicted), K))

#     for i, items in enumerate(items_ground_truth):
#         length = min(len(items), K)
#         test_matrix[i, :length] = 1
    
#     max_r = test_matrix
#     idcg = np.sum(max_r * 1. / np.log2(np.arange(2, K + 2)), axis=1)
#     dcg = items_predicted * (1. / np.log2(np.arange(2, K + 2)))
#     dcg = np.sum(dcg, axis=1)
#     idcg[idcg == 0.] = 1.
#     ndcg = dcg / idcg
#     ndcg[np.isnan(ndcg)] = 0.
    
#     return np.mean(ndcg)

# # wrapper function to get evaluation metrics
# def get_metrics(model, edge_index, exclude_edge_indices):

#     ratings = torch.matmul(model.emb_users.weight, model.emb_items.weight.T)

#     for exclude_edge_index in exclude_edge_indices:
#         user_pos_items = get_user_items(exclude_edge_index)
#         exclude_users = []
#         exclude_items = []
#         for user, items in user_pos_items.items():
#             exclude_users.extend([user] * len(items))
#             exclude_items.extend(items)
#         ratings[exclude_users, exclude_items] = -1024

#     # get the top k recommended items for each user
#     _, top_K_items = torch.topk(ratings, k=K)

#     # get all unique users in evaluated split
#     users = edge_index[0].unique()

#     test_user_pos_items = get_user_items(edge_index)

#     # convert test user pos items dictionary into a list
#     test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

#     # determine the correctness of topk predictions
#     items_predicted = []
#     for user in users:
#         ground_truth_items = test_user_pos_items[user.item()]
#         label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
#         items_predicted.append(label)

#     recall = compute_recall_at_k(test_user_pos_items_list, items_predicted)
#     ndcg = compute_ndcg_at_k(test_user_pos_items_list, items_predicted)

#     return recall, ndcg

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

def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        final_user_emb, final_item_emb = model(batch.edge_index)
        initial_user_emb, initial_item_emb = model.user_embedding.weight, model.item_embedding.weight

        user_indices = batch.edge_index[0, batch.edge_index[0] < num_users]
        pos_item_indices = batch.edge_index[1, batch.edge_index[1] >= num_users] - num_users
        neg_item_indices = sample_negative(pos_item_indices, model.num_items, device)
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
        
    return total_loss / len(train_loader)

def evaluate(model, test_data, device, k=20):
    model.eval()
    
    all_ndcg = []
    all_recall = []
    test_loss = 0
    with torch.no_grad():
        test_data = test_data.to(device)

        final_user_emb, final_item_emb = model(test_data.edge_index)
        initial_user_emb, initial_item_emb = model.user_embedding.weight, model.item_embedding.weight
        
        user_indices, pos_item_indices = test_data.edge_index[0], test_data.edge_index[1]
        neg_item_indices = sample_negative(test_data.edge_index, model.num_users, model.num_items, device)
        print("in the evaluate loop")
        # User Embeddings
        final_user_emb, initial_user_emb = final_user_emb[user_indices], initial_user_emb[user_indices]
        # Positive Sampling
        final_pos_item_emb, initial_pos_item_emb = final_item_emb[pos_item_indices], initial_item_emb[pos_item_indices]
        # Negative Sampling
        final_neg_item_emb, initial_neg_item_emb = final_item_emb[neg_item_indices], initial_item_emb[neg_item_indices]

        test_loss += bpr_loss(final_user_emb, initial_user_emb, 
                                final_pos_item_emb, initial_pos_item_emb, 
                                final_neg_item_emb, initial_neg_item_emb).item()

        # Compute scores for all couple user items in the loader
        scores = torch.mul(final_user_emb, final_pos_item_emb).sum(dim=-1)
        
        # Get top-k recommendations
        _, top_k_indices = torch.topk(scores, k=k)
        
        # # Compute NDCG and Recall
        # for i, user in enumerate(batch.edge_index[0].unique()):
        #     user_interactions = batch.edge_index[1][batch.edge_index[0] == user]
        #     recommended_items = top_k_indices[i]
            
        #     relevance = torch.isin(recommended_items, user_interactions).float()
        #     ideal_relevance = torch.ones_like(relevance)
            
        #     ndcg = ndcg_score([relevance.cpu().numpy()], [ideal_relevance.cpu().numpy()])
        #     recall = (relevance.sum() / len(user_interactions)).item()
            
        #     all_ndcg.append(ndcg)
        #     all_recall.append(recall)
    
    return test_loss#, np.mean(all_ndcg), np.mean(all_recall)

def train_model(model, train_loader, val_loader, test_loader, device, epochs=1, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # best_ndcg = 0
    # best loss equal to inf
    best_loss = float('inf')

    for epoch in range(epochs):
        loss = train(model, optimizer, train_loader, device)
        # val_ndcg, val_recall = evaluate(model, val_loader, device)
        val_loss = evaluate(model, val_loader, device)
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val NDCG@10: {val_ndcg:.4f}, Val Recall@10: {val_recall:.4f}')
        print(f'Epoch: {epoch:03d}, Loss Train: {loss:.4f}, Loss Val: {val_loss:.4f}')
        # scheduler.step(val_ndcg)
        scheduler.step(val_loss)
        
        # if val_ndcg > best_ndcg:
        if val_loss < best_loss:
            # best_ndcg = val_ndcg
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    # test_ndcg, test_recall = evaluate(model, test_loader, device)
    test_loss = evaluate(model, test_loader, device)
    # print(f'Test NDCG@20: {test_ndcg:.4f}, Test Recall@20: {test_recall:.4f}')
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
    for batch in train_loader:
        print(batch.edge_index[1].max(), batch.edge_index[1].min())
        break
    num_users, num_items = data_handler.get_num_users_items()
    model = LightGCN(num_users, num_items, dim_h=64, num_layers=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    trained_model = train_model(model, train_loader, val_data, test_data, device)