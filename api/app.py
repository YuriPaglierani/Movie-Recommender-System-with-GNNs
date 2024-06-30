from flask import Flask, request, jsonify
from models.light_gcn import LightGCN
from data.dataset_handler import load_movielens_data, get_movie_genres
from utils.train_test import train_model
import torch
import numpy as np

app = Flask(__name__)

# Load data
users, movies, ratings = load_movielens_data()

# Create graph
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user_id_map = {id: i for i, id in enumerate(user_ids)}
movie_id_map = {id: i + len(user_ids) for i, id in enumerate(movie_ids)}

edge_index = torch.tensor([
    [user_id_map[u] for u in ratings['userId']] + [movie_id_map[m] for m in ratings['movieId']],
    [movie_id_map[m] for m in ratings['movieId']] + [user_id_map[u] for u in ratings['userId']]
], dtype=torch.long)

# Split data
np.random.seed(42)
mask = np.random.rand(len(ratings)) < 0.8
train_ratings = ratings[mask]
test_ratings = ratings[~mask]

test_edge_index = torch.tensor([
    [user_id_map[u] for u in test_ratings['userId']],
    [movie_id_map[m] for m in test_ratings['movieId']]
], dtype=torch.long)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightGCN(len(user_ids), len(movie_ids), embedding_dim=64, num_layers=3).to(device)

# Train model
model = train_model(model, edge_index, test_edge_index, len(user_ids), len(movie_ids), device, num_clusters=100)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    num_recommendations = request.json.get('num_recommendations', 5)
    
    if user_id not in user_id_map:
        return jsonify({'error': 'User not found'}), 404
    
    user_embedding = model.get_embedding(user_indices=torch.tensor([user_id_map[user_id]]).to(device))
    item_embeddings = model.get_embedding(item_indices=torch.arange(len(movie_ids)).to(device))
    
    scores = torch.matmul(user_embedding, item_embeddings.t())
    top_items = torch.topk(scores, num_recommendations).indices.cpu().numpy()
    
    recommendations = [list(movie_id_map.keys())[i] for i in top_items[0]]
    recommended_movies = movies[movies['movieId'].isin(recommendations)]
    
    return jsonify({'recommendations': recommended_movies[['movieId', 'title', 'genres']].to_dict('records')})

@app.route('/add_user', methods=['POST'])
def add_user():
    genres = request.json['genres']
    
    # Create a new user embedding based on genre preferences
    genre_embeddings = model.get_embedding(item_indices=torch.arange(len(movie_ids)).to(device))
    genre_mask = movies['genres'].str.contains('|'.join(genres))
    genre_movie_embeddings = genre_embeddings[genre_mask]
    new_user_embedding = genre_movie_embeddings.mean(dim=0, keepdim=True)
    
    # Get recommendations for the new user
    scores = torch.matmul(new_user_embedding, genre_embeddings.t())
    top_items = torch.topk(scores, 10).indices.cpu().numpy()
    
    recommendations = [list(movie_id_map.keys())[i] for i in top_items[0]]
    recommended_movies = movies[movies['movieId'].isin(recommendations)]
    
    return jsonify({
        'recommendations': recommended_movies[['movieId', 'title', 'genres']].to_dict('records')
    })

@app.route('/movie_info/<int:movie_id>', methods=['GET'])
def movie_info(movie_id):
    if movie_id not in movie_id_map:
        return jsonify({'error': 'Movie not found'}), 404
    
    movie = movies[movies['movieId'] == movie_id].iloc[0]
    return jsonify({
        'movieId': int(movie['movieId']),
        'title': movie['title'],
        'genres': movie['genres'].split('|')
    })

@app.route('/genre_stats', methods=['GET'])
def genre_stats():
    genre_counts = movies['genres'].str.split('|', expand=True).stack().value_counts()
    return jsonify(genre_counts.to_dict())

@app.route('/user_stats/<int:user_id>', methods=['GET'])
def user_stats(user_id):
    if user_id not in user_id_map:
        return jsonify({'error': 'User not found'}), 404
    
    user_ratings = ratings[ratings['userId'] == user_id]
    stats = {
        'total_ratings': len(user_ratings),
        'average_rating': user_ratings['rating'].mean(),
        'favorite_genres': movies[movies['movieId'].isin(user_ratings['movieId'])]['genres'].str.split('|', expand=True).stack().value_counts().head(5).to_dict()
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)