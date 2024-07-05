import torch
# for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

def recommend_from_user(model, user_id, data_handler, excluded_train_items):

    user_id_map = data_handler.user_id_map
    movie_id_map = data_handler.movie_id_map
    movies = data_handler.movies

    user_index = user_id_map.get(user_id)
    
    if user_index is None:
        return {'error': 'Invalid user ID'}
    
    with torch.no_grad():
        user_embedding, item_embedding = model.get_embeddings(user_indices=torch.tensor([user_index]), 
                                                                      item_indices=torch.arange(len(movie_id_map)))
    
        scores = torch.matmul(user_embedding, item_embedding.t()).squeeze()
        
        # Get all movie indices sorted by score
        _, sorted_indices = torch.sort(scores, descending=True)
        sorted_indices 
        recommendations = []

        for idx in sorted_indices:
            if idx.item() in excluded_train_items:
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

def recommend_from_movie(model, movie_id, data_handler, excluded_train_users):
    user_id_map = data_handler.user_id_map
    movie_id_map = data_handler.movie_id_map

    movie_index = movie_id_map.get(movie_id) - len(user_id_map)
    
    if movie_index is None:
        return {'error': 'Invalid movie ID'}
    
    with torch.no_grad():
        user_embedding, item_embedding = model.get_embeddings(user_indices=torch.arange(len(user_id_map)), 
                                                                      item_indices=torch.tensor([movie_index]))
    
        scores = torch.matmul(user_embedding, item_embedding.t()).squeeze()
        
        # Get all user indices sorted by score
        _, sorted_indices = torch.sort(scores, descending=True)
        
        top_users = []
        for idx in sorted_indices:
            if idx.item() in excluded_train_users:
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
    # if best_model.pth is in the same directory as this file
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    user_ids = data_handler.user_id_map.keys()
    user_id = list(user_ids)[0]
    movie_id = list(data_handler.movie_id_map.keys())[0]

    train_data, _, _ = data_handler.get_datasets()
    train_edge_items = train_data.edge_index[1, train_data.edge_index[0, :] == 0] - len(user_ids) # take items of user 0
    train_edge_users = train_data.edge_index[0, train_data.edge_index[1, :]==0]
    print(recommend_from_user(model, user_id, data_handler, train_edge_items))
    print(recommend_from_movie(model, movie_id, data_handler, train_edge_users))

