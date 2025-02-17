import os
import requests
import zipfile
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

# if you want to track the memory management you can uncomment the lines with @profile
from memory_profiler import profile

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = "data/movielens-25m"

# for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

def download_and_extract_dataset() -> None:
    """
    Downloads and extracts the MovieLens 25M dataset.
    
    This function creates the data directory if it doesn't exist,
    downloads the dataset zip file, extracts only the 'movies.csv' 
    and 'ratings.csv' files, and removes the zip file after extraction.
    """

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    zip_path = os.path.join(DATA_DIR, "ml-25m.zip")
    
    print("Downloading MovieLens 25M dataset...")
    response = requests.get(MOVIELENS_25M_URL)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract only the required files
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        # for this project, we only need the 'movies.csv' and 'ratings.csv' files
        for file in ["ml-25m/movies.csv", "ml-25m/ratings.csv"]:

            zip_ref.extract(file, DATA_DIR)
            extracted_file_path = os.path.join(DATA_DIR, file)
            new_file_path = os.path.join(DATA_DIR, os.path.basename(file))
            os.rename(extracted_file_path, new_file_path)
    
    # Remove the zip file and the intermediate folder
    os.remove(zip_path)
    intermediate_folder = os.path.join(DATA_DIR, "ml-25m")

    if os.path.exists(intermediate_folder):
        os.rmdir(intermediate_folder)
    
    print("Dataset downloaded and extracted successfully.")

class MovieLensDataHandler:
    """
    Handles the MovieLens dataset processing and preparation for graph-based learning.
    
    This class is responsible for loading the MovieLens data, preprocessing it,
    creating graph structures, and preparing data loaders for training and evaluation.
    
    """
    # @profile
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load only necessary columns and filter ratings, we only consider ratings >= 4
        self.ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
        self.ratings = self.ratings[self.ratings['rating'] >= 4]
        
        # Load only necessary columns from movies
        self.movies = pd.read_csv(movies_path, usecols=['movieId', 'title'])
        
        self.num_users = self.ratings['userId'].nunique()
        self.num_movies = self.ratings['movieId'].nunique()
        
        # Create user and item mappings
        self.user_id_map = {id: i for i, id in enumerate(self.ratings['userId'].unique())}
        self.id_user_map = {i: id for id, i in self.user_id_map.items()}
        self.movie_id_map = {id: i+self.num_users for i, id in enumerate(self.ratings['movieId'].unique())}
        self.id_movie_map = {i+self.num_users: id for id, i in self.movie_id_map.items()}
        self._preprocess()

    # @profile
    def _preprocess(self) -> None:
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
        # Free up memory
        del self.ratings

        edge_index_np = np.vstack((user_idx, movie_idx))
        self.edge_index = torch.from_numpy(edge_index_np).long()
        self.edge_index = to_undirected(self.edge_index)
    
    # @profile
    def get_datasets(self, train_size: float = 0.9) -> Tuple[Data, Data, Data]:
        """
        Splits the data into train, validation, and test sets.
        
        Args:
            train_size (float): Proportion of data to use for training.
        
        Returns:
            Tuple: (train_dataset, val_dataset, test_dataset) as PyG Data objects.
        """

        indexes_path = "data/indexes"

        val_index_file = "val_indices.npy"
        test_index_file = "test_indices.npy"
        
        num_interactions = self.edge_index.shape[1]
        all_indices = np.arange(num_interactions)

        if not os.path.exists(indexes_path):
            print("Splitting data...")
            
            
            train_indices, val_test_indices = train_test_split(all_indices, train_size=train_size, shuffle=True)
            val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, shuffle=True)

            # sort the indices
            train_indices.sort()
            val_indices.sort()
            test_indices.sort()

            print("Saving indices...")
            print(train_indices)
            self._save_indices(val_indices, test_indices, indexes_path, val_index_file, test_index_file)

        else:
            print("Loading preprocessed data...")
            train_indices, val_indices, test_indices = self._load_from_indices(indexes_path, num_interactions, val_index_file, test_index_file) 

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

    def _load_from_indices(self, indexes_path: str, num_interactions: int, val_index_file: str, test_index_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the validation, and test indices from files and infers what are the training indices from num_interactions.
        
        Args:
            indexes_path (str): Path to load the indices from.
            num_interactions (int): Number of interactions in the dataset.
            val_index_file (str): Path to load the validation indices from.
            test_index_file (str): Path to load the test indices from.
        
        Returns:
            Tuple: (train_indices, val_indices, test_indices) as np.ndarray objects.
        """

        if not os.path.exists(indexes_path):
            raise FileNotFoundError("Indexes path not found. Please preprocess the data first.")
        
        val_idx = os.path.join(indexes_path, val_index_file)
        test_idx = os.path.join(indexes_path, test_index_file)
        
        val_indices = np.sort(np.load(val_idx))
        test_indices = np.sort(np.load(test_idx))

        all_indices = np.arange(num_interactions)
        val_test_indices = np.concatenate((val_indices, test_indices))
        train_indices = np.setdiff1d(all_indices, val_test_indices)

        # check that all the indices are sorted
        assert np.all(np.diff(train_indices) > 0)
        assert np.all(np.diff(val_indices) > 0)
        assert np.all(np.diff(test_indices) > 0)

        return train_indices, val_indices, test_indices
    
    def _save_indices(self, val_indices: np.ndarray, test_indices: np.ndarray, 
                     indexes_path: str, val_index_file: str, test_index_file: str) -> None:
        """
        Saves the validation, and test indices to files.
        
        Args:
            val_indices (np.ndarray): Indices for the validation set.
            test_indices (np.ndarray): Indices for the test set.
            val_index_file (str): Path to save the validation indices.
            test_index_file (str): Path to save the test indices.
        """
        if not os.path.exists(indexes_path):
            os.makedirs(indexes_path)

        val_idx = os.path.join(indexes_path, val_index_file)
        test_idx = os.path.join(indexes_path, test_index_file)

        np.save(val_idx, val_indices)
        np.save(test_idx, test_indices)

    # @profile
    def get_data_training(self, num_train_clusters: int = 100) -> Tuple[DataLoader, Data, Data]:
        """
        Creates Dataloader for train set, and return Data for validation, and test sets.
        
        This method uses ClusterData and DataLoader for efficient
        handling of large graphs.
        
        Args:
            num_train_clusters (int): Number of clusters for training data.
        
        Returns:
            Tuple: (train_loader, val_dataset, test_dataset)
        """

        print("Creating dataloaders...")
        train_dataset, val_dataset, test_dataset = self.get_datasets()

        cluster_train = ClusterData(train_dataset, num_parts=num_train_clusters)
        del train_dataset

        train_l = []
        for cluster in cluster_train:
            right_cluster = Data(edge_index=cluster.n_id[cluster.edge_index], 
                                num_nodes=self.num_users + self.num_movies)
            
            right_cluster.n_id = torch.arange(self.num_users + self.num_movies, device=self.device)
            train_l.append(right_cluster)
        
        del cluster_train
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
    print("Number of users:", data_handler.get_num_users_items()[0])
    print("Number of items:", data_handler.get_num_users_items()[1])
    print("Number of relevant interactions:", data_handler.edge_index.shape[1])

    train_loader, val_data, test_data = data_handler.get_data_training()

    # Print an iteration over the train loader
    for batch in train_loader:
        if batch.edge_index.numel() == 0:
            print("Empty batch detected, skipping...")
            continue
        # print(batch)