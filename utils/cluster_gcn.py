import torch
import numpy as np
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans

def cluster_graph(edge_index, num_clusters, num_users, num_items):
    edge_index = to_undirected(edge_index)
    
    adj = torch.zeros((num_users + num_items, num_users + num_items))
    adj[edge_index[0], edge_index[1]] = 1
    
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(adj.cpu().numpy())
    
    return cluster_labels

def get_cluster_loader(edge_index, cluster_labels, batch_size, num_users, num_items):
    num_clusters = len(np.unique(cluster_labels))
    
    def sample_cluster():
        cluster = np.random.choice(num_clusters)
        node_mask = torch.from_numpy(cluster_labels == cluster)
        user_mask = node_mask[:num_users]
        item_mask = node_mask[num_users:]
        
        users = torch.where(user_mask)[0]
        items = torch.where(item_mask)[0] + num_users
        
        batch_nodes = torch.cat([users, items])
        
        edge_mask = torch.isin(edge_index[0], batch_nodes) & torch.isin(edge_index[1], batch_nodes)
        batch_edge_index = edge_index[:, edge_mask]
        
        node_id_map = {int(node_id): idx for idx, node_id in enumerate(batch_nodes)}
        batch_edge_index = torch.tensor([[node_id_map[int(node_id)] for node_id in batch_edge_index[0]],
                                         [node_id_map[int(node_id)] for node_id in batch_edge_index[1]]])
        
        return batch_edge_index, len(batch_nodes)
    
    return sample_cluster