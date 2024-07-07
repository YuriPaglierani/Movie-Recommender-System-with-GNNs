import os 
import pandas as pd
import requests
import zipfile
from typing import Tuple

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = "data/movielens-25m"

def download_and_extract_dataset_trial() -> None:
    """
    Downloads and extracts the MovieLens 25M dataset.
    
    This function creates the data directory if it doesn't exist,
    downloads the dataset zip file, extracts all files,
    and removes the zip file after extraction.

    This function is only used for demonstration purposes, and must not be imported in other files
    """

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    zip_path = os.path.join(DATA_DIR, "ml-25m.zip")
    
    # Download the dataset
    print("Downloading MovieLens 25M dataset...")
    response = requests.get(MOVIELENS_25M_URL)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract all files
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Remove the zip file
    os.remove(zip_path)
    
    print("Dataset downloaded and extracted successfully.")

# Run the function to download and extract the dataset
download_and_extract_dataset_trial()

# perform EDA on the dataset, analyze all the csv files 
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the MovieLens 25M dataset.
    
    Returns:
        pd.DataFrame: Ratings data.
        pd.DataFrame: Movies data.
        pd.DataFrame: Tags data.
    """
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ml-25m", "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "ml-25m", "movies.csv"))
    tags = pd.read_csv(os.path.join(DATA_DIR, "ml-25m", "tags.csv"))
    
    return ratings, movies, tags

ratings, movies, tags = load_data()

# Display the first few rows of the ratings data
print("Ratings data:")
print(ratings.head())

# Display the first few rows of the movies data 
print("\nMovies data:")
print(movies.head())

# Display the first few rows of the tags data
print("\nTags data:")
print(tags.head())

# Display the number of unique users and movies in the ratings data
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
print(f"\nNumber of unique users: {num_users}")
print(f"Number of unique movies: {num_movies}")

# Display the number of unique genres in the movies data
unique_genres = set(movies['genres'].str.split('|').sum())
num_genres = len(unique_genres)
print(f"\nNumber of unique genres: {num_genres}")

# Display the number of unique tags in the tags data
num_tags = tags['tag'].nunique()
print(f"\nNumber of unique tags: {num_tags}")

# Display the number of ratings per user
ratings_per_user = ratings.groupby('userId').size()
print("\nRatings per user:")
print(ratings_per_user.describe())

# Display the number of ratings per movie
ratings_per_movie = ratings.groupby('movieId').size()
print("\nRatings per movie:")
print(ratings_per_movie.describe())

# Compute the average degree of the movie graph
average_degree = ratings_per_movie.mean()
print(f"\nAverage degree of the movie graph: {average_degree:.2f}")

# Compute the number of ratings that are greater or equal than 4, compute also the fraction wrt the total number of ratings
num_positive_ratings = (ratings['rating'] >= 4).sum()
fraction_positive_ratings = num_positive_ratings / len(ratings)
print(f"\nNumber of positive ratings: {num_positive_ratings}")
print(f"Fraction of positive ratings: {fraction_positive_ratings:.2f}")

import shutil
shutil.rmtree('data/movielens-25m')