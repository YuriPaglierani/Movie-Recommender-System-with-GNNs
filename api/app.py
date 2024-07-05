from flask import Flask, request, jsonify
import torch
import os
from models.light_gcn import LightGCN
from data.dataset_handler import MovieLensDataHandler
from utils.recommend import recommend_from_user

# for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

app = Flask(__name__)

# Load your trained model and data handler
DATA_DIR = "data/movielens-25m"

data_handler = MovieLensDataHandler(os.path.join(DATA_DIR, "ratings.csv"), os.path.join(DATA_DIR, "movies.csv"))
data_handler.preprocess()

num_users, num_items = data_handler.get_num_users_items()
model = LightGCN(num_users, num_items)  # Replace with your actual model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

@app.route('/user_to_movie', methods=['POST'])
def user_to_movie():
    user_id = request.json['user_id']
    return jsonify(recommend_from_user(model, user_id, data_handler))

@app.route('/movie_to_user', methods=['POST'])
def movie_to_user():
    movie_id = request.json['movie_id']
    
    # Convert movie_id to the format used by your model
    movie_index = data_handler.movie_id_map.get(movie_id)
    
    if movie_index is None:
        return jsonify({'error': 'Invalid movie ID'}), 400
    
    with torch.no_grad():
        user_embedding, _ = model.get_final_embeddings(user_indices=torch.arange(data_handler.num_users))
        _, item_embedding = model.get_final_embeddings(item_indices=torch.tensor([movie_index]))
        
        scores = torch.matmul(user_embedding, item_embedding.t()).squeeze()
        
        # Get top 10 user recommendations
        top_scores, top_indices = torch.topk(scores, 10)
        
        top_users = []
        for idx, score in zip(top_indices, top_scores):
            user_id = list(data_handler.user_id_map.keys())[list(data_handler.user_id_map.values()).index(idx.item())]
            top_users.append({
                'user_id': user_id,
                'score': score.item()
            })
    
    return jsonify({'top_users': top_users})

@app.route('/get_embeddings', methods=['GET'])
def get_embeddings():
    with torch.no_grad():
        user_embedding, item_embedding = model.get_final_embeddings()
        return jsonify({
            'user_embedding': user_embedding.tolist(),
            'item_embedding': item_embedding.tolist()
        })

if __name__ == '__main__':
    app.run(debug=True)