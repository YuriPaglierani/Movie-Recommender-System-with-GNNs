from flask import Flask, request, jsonify
import torch
import os
from models.light_gcn import LightGCN
from data.dataset_handler import MovieLensDataHandler
from utils.recommend import recommend_from_user, recommend_from_movie

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

num_users, num_items = data_handler.get_num_users_items()
model = LightGCN(num_users, num_items)  # Replace with your actual model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

@app.route('/user_to_movie', methods=['POST'])
def user_to_movie():
    try:
        user_id = request.json['user_id']
        excluded_train_items = request.json['excluded_train_items']
        result = recommend_from_user(model, user_id, data_handler, excluded_train_items)
        return jsonify(result)
    except Exception as e:
        print(f"Error in /user_to_movie: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/movie_to_user', methods=['POST'])
def movie_to_user():
    try:
        movie_id = request.json['movie_id']
        excluded_train_users = request.json['excluded_train_users']
        result = recommend_from_movie(model, movie_id, data_handler, excluded_train_users)
        return jsonify(result)
    except Exception as e:
        print(f"Error in /movie_to_user: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_embeddings', methods=['GET'])
def get_embeddings():
    with torch.no_grad():
        user_embedding, item_embedding = model.get_final_embeddings()
        return jsonify({
            'user_embedding': user_embedding.tolist(),
            'item_embedding': item_embedding.tolist()
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
