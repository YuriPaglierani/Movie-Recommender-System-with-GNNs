import torch
from typing import Tuple, Dict, List

def cantor_hash_pair(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Cantor pairing function for two tensors.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Hashed values using Cantor pairing function.
    """
    return ((x + y) * (x + y + 1)) // 2 + y

def get_user_items(edge_index: torch.Tensor) -> Dict[int, List[int]]:
    """
    Create a dictionary mapping users to their interacted items.

    Args:
        edge_index (torch.Tensor): Edge indices of the graph.

    Returns:
        dict[int, list[int]]: Dictionary mapping user IDs to lists of item IDs.
    """
    user_items = dict()
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_items:
            user_items[user] = []
        user_items[user].append(item)
    return user_items

def is_in_feasible(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Find elements in A that are not in B using Cantor hashing.

    Args:
        A (torch.Tensor): First input tensor of shape (2, n).
        B (torch.Tensor): Second input tensor of shape (2, m).

    Returns:
        torch.Tensor: Subset of A containing elements not in B.
    """

    A_hash = cantor_hash_pair(A[0], A[1])
    B_hash = cantor_hash_pair(B[0], B[1])
    A_hash_sorted, indices_A = A_hash.sort()
    B_hash_sorted, _ = B_hash.sort()

    mask = ~torch.isin(A_hash_sorted, B_hash_sorted)
    indices_A = indices_A[mask]
    return A[:, indices_A]

def sample_negative(pos_idx: torch.Tensor, num_items: int, device: torch.device) -> torch.Tensor:
    """
    Sample negative items for each positive item.

    Note: This is a simplified negative sampling function and may not be suitable for dense graphs.

    Args:
        pos_idx (torch.Tensor): Indices of positive items.
        num_items (int): Total number of items.
        device (torch.device): Device to create the tensor on.

    Returns:
        torch.Tensor: Indices of sampled negative items.
    """
    
    neg_item_index = torch.randint(0, num_items, 
                                  (pos_idx.shape[0], ), device=device) 
     
    return neg_item_index

def get_triplets_indices(edge_index: torch.Tensor, num_users: int, num_items: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate triplets (user, positive item, negative item) from edge indices.

    Args:
        edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
        num_users (int): Number of users in the graph.
        num_items (int): Number of items in the graph.
        device (torch.device): Device to create tensors on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: User indices, positive item indices, and negative item indices.
    """

    user_indices = edge_index[0, edge_index[0] < num_users]
    pos_item_indices = edge_index[1, edge_index[1] >= num_users] - num_users
    neg_item_indices = sample_negative(pos_item_indices, num_items, device)
        
    return user_indices, pos_item_indices, neg_item_indices
    