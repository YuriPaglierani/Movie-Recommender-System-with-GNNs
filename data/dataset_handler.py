import os
import requests
import zipfile
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import ClusterData, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = "data/movielens-25m"

def download_and_extract_dataset() -> None:
    """
    Downloads and extracts the MovieLens 25M dataset.
    
    This function creates the data directory if it doesn't exist,
    downloads the dataset zip file, extracts its contents,
    and removes the zip file after extraction.
    """

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
    
class MovieLensDataHandler:
    """
    Handles the MovieLens dataset processing and preparation for graph-based learning.
    
    This class is responsible for loading the MovieLens data, preprocessing it,
    creating graph structures, and preparing data loaders for training and evaluation.
    
    """

    def __init__(self, ratings_path: str, movies_path: str):
        """
        Initializes the MovieLensDataHandler.
        
        Args:
            ratings_path (str): Path to the ratings CSV file.
            movies_path (str): Path to the movies CSV file.
        """
        self.ratings_path: str = ratings_path
        self.movies_path: str = movies_path
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ratings: pd.DataFrame
        self.movies: pd.DataFrame
        self.num_users: int
        self.num_movies: int
        self.user_id_map: Dict[int, int]
        self.movie_id_map: Dict[int, int]
        self.edge_index: torch.Tensor

        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            print("Dataset not found. Downloading...")
            download_and_extract_dataset()

        print("Loading dataset...")
        
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        
        # Load only necessary columns and filter ratings
        self.ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
        self.ratings = self.ratings[self.ratings['rating'] >= 4]
        
        # Load only necessary columns from movies
        self.movies = pd.read_csv(movies_path, usecols=['movieId', 'genres'])
        
        self.num_users = self.ratings['userId'].nunique()
        self.num_movies = self.ratings['movieId'].nunique()
        
        # Create user and item mappings
        self.user_id_map = {id: i for i, id in enumerate(self.ratings['userId'].unique())}
        self.movie_id_map = {id: i+self.num_users for i, id in enumerate(self.ratings['movieId'].unique())}
        
    def preprocess(self) -> None:
        """
        Preprocesses the loaded data.
        
        This method converts user and movie IDs to sequential integers,
        creates an edge index for the graph, and frees up memory by
        deleting the original ratings dataframe.
        """

        print("Preprocessing data...")

        # Convert user and movie IDs to sequential integers
        user_idx = self.ratings['userId'].map(self.user_id_map).values
        movie_idx = self.ratings['movieId'].map(self.movie_id_map).values
        edge_index_np = np.vstack((user_idx, movie_idx))
        self.edge_index = torch.from_numpy(edge_index_np).long()
        self.edge_index = to_undirected(self.edge_index)

        # Free up memory
        del self.ratings
        
    def split_data(self, train_size: float = 0.9) -> Tuple[Data, Data, Data]:
        """
        Splits the data into train, validation, and test sets.
        
        Args:
            train_size (float): Proportion of data to use for training.
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset) as PyG Data objects.
        """

        print("Splitting data...")
        num_interactions = self.edge_index.shape[1]
        all_indices = np.arange(num_interactions)
        
        train_indices, val_test_indices = train_test_split(all_indices, train_size=train_size, shuffle=True)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, shuffle=True)
        
        train_edges = self.edge_index[:, train_indices].contiguous()
        val_edges = self.edge_index[:, val_indices].contiguous()
        test_edges = self.edge_index[:, test_indices].contiguous()

        train_dataset = Data(edge_index=train_edges, 
                               num_nodes=self.num_users + self.num_movies).to(self.device)
        train_dataset.n_id = torch.arange(self.num_users + self.num_movies, device=self.device)
    
        val_dataset = Data(edge_index=val_edges, 
                               num_nodes=self.num_users + self.num_movies).to(self.device)
        val_dataset.n_id = torch.arange(self.num_users + self.num_movies, device=self.device)

        test_dataset = Data(edge_index=test_edges, 
                               num_nodes=self.num_users + self.num_movies).to(self.device)
        test_dataset.n_id = torch.arange(self.num_users + self.num_movies, device=self.device)

        return train_dataset, val_dataset, test_dataset

    def get_data(self, num_train_clusters: int = 100) -> Tuple[DataLoader, Data, Data]:
        """
        Creates ClusterLoader for train set, and return Data for validation, and test sets.
        
        This method uses ClusterData and DataLoader for efficient
        handling of large graphs.
        
        Args:
            num_train_clusters (int): Number of clusters for training data.
        
        Returns:
            tuple: (train_loader, val_dataset, test_dataset)
        """

        print("Creating dataloaders...")
        train_dataset, val_dataset, test_dataset = self.split_data()

        cluster_train = ClusterData(train_dataset, num_parts=num_train_clusters)
        del train_dataset

        train_l = []
        for cluster in cluster_train:
            right_cluster = Data(edge_index=cluster.n_id[cluster.edge_index], 
                                num_nodes=self.num_users + self.num_movies)
            
            right_cluster.n_id = torch.arange(self.num_users + self.num_movies, device=self.device)
            train_l.append(right_cluster)
        
        del cluster_train
        # batched_train_set = Batch.from_data_list(train_l)
        train_loader = DataLoader(train_l, batch_size=1, shuffle=True)
        del train_l
        
        return train_loader, val_dataset, test_dataset
    
    def get_num_users_items(self) -> Tuple[int, int]:
        """
        Returns the number of unique users and items in the dataset.
        
        Returns:
            tuple: (num_users, num_items)
        """
        
        return len(self.user_id_map), len(self.movie_id_map)

if __name__ == "__main__":
    data_handler = MovieLensDataHandler(os.path.join(DATA_DIR, "ratings.csv"), os.path.join(DATA_DIR, "movies.csv"))
    data_handler.preprocess()

    print("Number of users:", data_handler.get_num_users_items()[0])
    print("Number of items:", data_handler.get_num_users_items()[1])
    print("Number of relevant interactions:", data_handler.edge_index.shape[1])

    data_handler.split_data()
    train_loader, val_data, test_data = data_handler.get_data()

    # Print an iteration over the train loader
    for batch in train_loader:
        if batch.edge_index.numel() == 0:
            print("Empty batch detected, skipping...")
            continue
        # print(batch)
