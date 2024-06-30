import os
import requests
import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.model_selection import train_test_split

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = "data/movielens-25m"

def download_and_extract_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    zip_path = os.path.join(DATA_DIR, "ml-25m.zip")
    
    # Download the dataset
    print("Downloading MovieLens 25M dataset...")
    response = requests.get(MOVIELENS_25M_URL)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    extracted_folder = os.path.join(DATA_DIR, "ml-25m")
    for file in os.listdir(extracted_folder):
        os.rename(os.path.join(extracted_folder, file), os.path.join(DATA_DIR, file))
    os.rmdir(extracted_folder)

    # Remove the zip file
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully.")

class MovieLensDataset(Dataset):
    def __init__(self, user_movie_pairs):
        self.user_movie_pairs = user_movie_pairs

    def __len__(self):
        return len(self.user_movie_pairs)

    def __getitem__(self, idx):
        user, movie = self.user_movie_pairs[idx]
        return user, movie
    
class MovieLensDataHandler:
    def __init__(self, ratings_path, movies_path):
        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            print("Dataset not found. Downloading...")
            download_and_extract_dataset()

        print("Loading dataset...")
        
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load only necessary columns and filter ratings
        self.ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
        self.ratings = self.ratings[self.ratings['rating'] >= 4]
        
        # Load only necessary columns from movies
        self.movies = pd.read_csv(movies_path, usecols=['movieId', 'genres'])
        
        # Create user and item mappings
        self.user_id_map = {id: i for i, id in enumerate(self.ratings['userId'].unique())}
        self.movie_id_map = {id: i for i, id in enumerate(self.ratings['movieId'].unique())}
        
    def preprocess(self):
        print("Preprocessing data...")

        # Convert user and movie IDs to sequential integers
        user_idx = self.ratings['userId'].map(self.user_id_map).values
        movie_idx = self.ratings['movieId'].map(self.movie_id_map).values
        
        # Create edge index using numpy first, then convert to tensor
        edge_index_np = np.vstack((user_idx, movie_idx))
        self.edge_index = torch.from_numpy(edge_index_np).long()
        
        # Create graph data
        self.graph_data = Data(edge_index=self.edge_index, 
                               num_nodes=len(self.user_id_map) + len(self.movie_id_map))
        
        # Free up memory
        del self.ratings
        
    def split_data(self, test_size=0.2, val_size=0.1):
        print("Splitting data...")
        num_interactions = self.edge_index.shape[1]
        all_indices = np.arange(num_interactions)
        
        train_val_indices, test_indices = train_test_split(all_indices, test_size=test_size, shuffle=True)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, shuffle=True)
        
        self.train_dataset = self._create_dataset(train_indices)
        self.val_dataset = self._create_dataset(val_indices)
        self.test_dataset = self._create_dataset(test_indices)
        
    def _create_dataset(self, indices):
        return MovieLensDataset(self.edge_index[:, indices].t())
    
    def get_dataloaders(self, batch_size=1024, num_clusters=10):
        print("Creating dataloaders...")
        cluster_data = ClusterData(self.graph_data, num_parts=num_clusters)
        
        train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_genre_list(self):
        return self.movies['genres'].str.get_dummies('|').columns.tolist()
    
    def get_num_users_items(self):
        return len(self.user_id_map), len(self.movie_id_map)

if __name__ == "__main__":
    data_handler = MovieLensDataHandler(os.path.join(DATA_DIR, "ratings.csv"), os.path.join(DATA_DIR, "movies.csv"))
    data_handler.preprocess()

    print("Number of users:", data_handler.get_num_users_items()[0])
    print("Number of items:", data_handler.get_num_users_items()[1])
    print("Number of relevant interactions:", data_handler.edge_index.shape[1])
    print("Genre list:", data_handler.get_genre_list())

    data_handler.split_data()
    train_loader, val_loader, test_loader = data_handler.get_dataloaders()