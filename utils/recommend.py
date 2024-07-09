from typing import List, Dict, Any, Union, Optional
import torch
from visualizations import plot_recommendations, analyze_user_recommendations

# for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

def recommend_from_user(model: torch.nn.Module, user_id: int, data_handler: Any, excluded_train_items: Optional[List[int]] = None) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
    """
    Recommend movies for a given user.

    Args:
        model (torch.nn.Module): The trained recommendation model.
        user_id (int): The ID of the user for whom to recommend movies.
        data_handler (Any): The data handler containing user and movie data.
        excluded_train_items (Optional[List[int]]): List of item indices to exclude from recommendations. Optional.

    Returns:
        Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]: A dictionary containing recommendations or an error message.
    """

    user_id_map = data_handler.user_id_map
    movie_id_map = data_handler.movie_id_map
    movies = data_handler.movies

    user_index = user_id_map.get(user_id)
    
    if user_index is None:
        return {'error': 'Invalid user ID'}
    
    with torch.no_grad():
        user_embedding, item_embedding = model.get_embeddings(user_indices=torch.tensor([user_index]), 
                                                                      item_indices=torch.arange(len(movie_id_map)))
    
        user_embedding = user_embedding / torch.norm(user_embedding, dim=1, keepdim=True)
        item_embedding = item_embedding / torch.norm(item_embedding, dim=1, keepdim=True)
        
        scores = torch.matmul(user_embedding, item_embedding.t()).squeeze()
        
        _, sorted_indices = torch.sort(scores, descending=True)
        
        recommendations = []

        for idx in sorted_indices:
            if (not excluded_train_items is None) and idx.item() in excluded_train_items:
                continue

            movie_id = list(movie_id_map.keys())[list(movie_id_map.values()).index(idx.item()+len(user_id_map))]

            movie_info = movies[movies['movieId'] == movie_id].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'score': scores[idx].item()
            })
            
            if len(recommendations) == 10:
                break
    
    return {'recommendations': recommendations}

def recommend_from_movie(model: torch.nn.Module, movie_id: int, data_handler: Any, excluded_train_users: Optional[List[int]] = None) -> Dict[str, Union[str, List[Dict[str, Union[int, float]]]]]:
    """
    Recommend users for a given movie.

    Args:
        model (torch.nn.Module): The trained recommendation model.
        movie_id (int): The ID of the movie for which to recommend users.
        data_handler (Any): The data handler containing user and movie data.
        excluded_train_users (Optional[List[int]]): List of user indices to exclude from recommendations. Optional.

    Returns:
        Dict[str, Union[str, List[Dict[str, Union[int, float]]]]]: A dictionary containing top users or an error message.
    """
    
    user_id_map = data_handler.user_id_map
    movie_id_map = data_handler.movie_id_map

    movie_index = movie_id_map.get(movie_id)
    
    if movie_index is None:
        return {'error': 'Invalid movie ID'}
    
    movie_index -= len(user_id_map)  # Adjust index
    
    with torch.no_grad():
        user_embedding, item_embedding = model.get_embeddings(user_indices=torch.arange(len(user_id_map)), 
                                                              item_indices=torch.tensor([movie_index]))
    
        user_embedding = user_embedding / torch.norm(user_embedding, dim=1, keepdim=True)
        item_embedding = item_embedding / torch.norm(item_embedding, dim=1, keepdim=True)
        
        scores = torch.matmul(user_embedding, item_embedding.t()).squeeze()
        
        _, sorted_indices = torch.sort(scores, descending=True)
        
        top_users = []
        for idx in sorted_indices:
            if (excluded_train_users is not None) and idx.item() in excluded_train_users:
                continue
            user_id = list(user_id_map.keys())[list(user_id_map.values()).index(idx.item())]
            top_users.append({
                'user_id': user_id,
                'score': scores[idx].item()
            })
            
            if len(top_users) == 10:
                break
    
    return {'top_users': top_users}

if __name__ == '__main__':
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)    

    from data.dataset_handler import MovieLensDataHandler
    from models.light_gcn import LightGCN

    data_handler = MovieLensDataHandler('data/movielens-25m/ratings.csv', 'data/movielens-25m/movies.csv')
    num_users, num_items = data_handler.get_num_users_items()

    model = LightGCN(num_users, num_items)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    user_ids = list(data_handler.user_id_map.keys())
    suggested_user = user_ids[0]
    
    print(f"Please enter a user ID (suggested user: {suggested_user}):")
    user_id = int(input())

    train_data, _, _ = data_handler.get_datasets()
    train_edge_items = train_data.edge_index[1, train_data.edge_index[0, :] == data_handler.user_id_map[user_id]] - len(user_ids)
    
    recommendations = recommend_from_user(model, user_id, data_handler, train_edge_items)
    
    if 'error' in recommendations:
        print(recommendations['error'])
    else:
        print(f"Top 10 Recommendations for user {user_id}:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"{i}. {rec['title']} (Score: {rec['score']:.4f})")
        
        plot_recommendations(recommendations['recommendations'], user_id)

    analysis_plot = analyze_user_recommendations(model, user_id, data_handler)
    analysis_plot.show()