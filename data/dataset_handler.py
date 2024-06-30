import os
import requests
import zipfile
import pandas as pd

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
    
    # the content is in movielens-25m/ml-25m, take the folder ml-25m, move its content to the parent folder and remove the folder
    extracted_folder = os.path.join(DATA_DIR, "ml-25m")
    for file in os.listdir(extracted_folder):
        os.rename(os.path.join(extracted_folder, file), os.path.join(DATA_DIR, file))
    os.rmdir(extracted_folder)

    # Remove the zip file
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully.")

def load_movielens_data():

    if not os.path.exists(os.path.join(DATA_DIR, "movies.csv")):
        print("Dataset not found. Downloading...")
        download_and_extract_dataset()
    
    print("Loading dataset...")
    
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    users = pd.DataFrame({'userId': ratings['userId'].unique()})
    
    print("Dataset loaded successfully.")
    
    return users, movies, ratings

def get_movie_genres(movies):
    return movies['genres'].str.split('|').explode().unique().tolist()

if __name__ == "__main__":

    _, movies, _ = load_movielens_data()
    genres = get_movie_genres(movies)

    print("Movie genres:")
    print(genres)
