import torch
import torch.optim as optim
from torch_geometric.utils import structured_negative_sampling
from sklearn.metrics import roc_auc_score
import numpy as np
from utils.cluster_gcn import cluster_graph, get_cluster_loader

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    return loss

def train(model, optimizer, cluster_loader, num_users, num_items, device):
    model.train()
    total_loss = 0
    
    for _ in range(len(cluster_loader)):
        optimizer.zero_grad()
        
        batch_edge_index, batch_size = cluster_loader()
        batch_edge_index = batch_edge_index.to(device)
        
        user_emb, item_emb = model(batch_edge_index, batch_size)
        
        user, pos_item, neg_item = structured_negative_sampling(batch_edge_index, num_nodes=batch_size)
        
        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item - num_users]
        neg_item_emb = item_emb[neg_item - num_users]
        
        loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(cluster_loader)

def test(model, edge_index, test_edges, num_users, num_items, device):
    model.eval()
    
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
    
    user_emb = user_emb.cpu().numpy()
    item_emb = item_emb.cpu().numpy()
    
    test_users, test_items = test_edges
    test_users -= 1  # Adjust for 0-based indexing
    test_items -= (num_users + 1)  # Adjust for 0-based indexing and user count
    
    scores = np.sum(user_emb[test_users] * item_emb[test_items], axis=1)
    
    neg_users = np.random.randint(0, num_users, size=len(test_users))
    neg_items = np.random.randint(0, num_items, size=len(test_users))
    neg_scores = np.sum(user_emb[neg_users] * item_emb[neg_items], axis=1)
    
    labels = np.concatenate([np.ones_like(scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([scores, neg_scores])
    
    auc = roc_auc_score(labels, scores)
    
    return auc

def train_model(model, edge_index, test_edges, num_users, num_items, device, epochs=100, lr=0.001, weight_decay=1e-5, num_clusters=100):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Cluster the graph
    cluster_labels = cluster_graph(edge_index, num_clusters, num_users, num_items)
    cluster_loader = get_cluster_loader(edge_index, cluster_labels, num_clusters, num_users, num_items)
    
    best_auc = 0
    for epoch in range(epochs):
        loss = train(model, optimizer, cluster_loader, num_users, num_items, device)
        auc = test(model, edge_index, test_edges, num_users, num_items, device)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model