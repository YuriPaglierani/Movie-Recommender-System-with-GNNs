import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import requests
import os
import sys
import pandas as pd
import torch
import dash_bootstrap_components as dbc

# Set seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

# Import the necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.visualizations import create_user_item_graph, plot_user_item_graph, plot_embedding_tsne
from data.dataset_handler import MovieLensDataHandler
from models.light_gcn import LightGCN
from utils.recommend import recommend_from_user, recommend_from_movie

DATA_DIR = "data/movielens-25m"

data_handler = MovieLensDataHandler('data/movielens-25m/ratings.csv', 'data/movielens-25m/movies.csv')
train_data, _, _ = data_handler.get_datasets()
user_ids = data_handler.user_id_map.keys()
movies_id_map = data_handler.movie_id_map
movie_titles = data_handler.movies['title'].tolist()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1('Movie Recommender System'),
    
    dcc.RadioItems(
        id='mode-selection',
        options=[
            {'label': 'User to Movie', 'value': 'user_to_movie'},
            {'label': 'Movie to User', 'value': 'movie_to_user'}
        ],
        value='user_to_movie'
    ),
    
    html.Div(id='input-container'),
    
    html.Button('Get Recommendations', id='recommend-button'),
    html.Div(id='recommendations-output'),
    
    html.Div([
        html.H2('Recommender System Visualizations'),
        dcc.Dropdown(
            id='visualization-type',
            options=[
                {'label': 'User-Item Interaction Graph', 'value': 'user-item-graph'},
                {'label': 't-SNE Embedding Visualization', 'value': 'tsne-embedding'}
            ],
            value='user-item-graph'
        ),
        dcc.Graph(id='visualization-graph')
    ]),

    # Modal for selecting a movie
    dbc.Modal(
        [
            dbc.ModalHeader("Select a Movie"),
            dbc.ModalBody(
                dcc.Dropdown(
                    id='movie-dropdown',
                    options=[{'label': title, 'value': title} for title in movie_titles],
                    placeholder="Select a movie"
                )
            ),
            dbc.ModalFooter(
                dbc.Button("Select", id="select-movie-button", className="ml-auto")
            ),
        ],
        id="modal",
        is_open=False,
    )
])

@app.callback(
    Output('input-container', 'children'),
    Input('mode-selection', 'value')
)
def update_input(mode):
    if mode == 'user_to_movie':
        return dcc.Input(id='id-input', type='number', placeholder='Enter User ID')
    else:
        return html.Div([
            html.Button('Select a Movie', id='open-modal-button'),
            html.Div(id='selected-movie', style={'marginTop': '10px'})
        ])

@app.callback(
    Output('modal', 'is_open'),
    [Input('open-modal-button', 'n_clicks'), Input('select-movie-button', 'n_clicks')],
    [State('modal', 'is_open')]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('selected-movie', 'children'),
    Input('select-movie-button', 'n_clicks'),
    State('movie-dropdown', 'value')
)
def update_selected_movie(n, value):
    if n and value:
        return f'Selected Movie: {value}'
    return ''

@app.callback(
    Output('recommendations-output', 'children'),
    Input('recommend-button', 'n_clicks'),
    State('mode-selection', 'value'),
    State('id-input', 'value'),
    State('selected-movie', 'children')
)
def update_recommendations(n_clicks, mode, id_value, selected_movie):
    if n_clicks is None:
        return 'Enter an ID and click the button to get recommendations.'
    
    if mode == 'user_to_movie':
        if int(id_value) not in user_ids:
            return 'User ID not found in dataset.'
        
        excluded_train_items = train_data.edge_index[1, train_data.edge_index[0, :] == int(id_value)].tolist()
        response = requests.post('http://localhost:5000/user_to_movie', json={'user_id': int(id_value), 'excluded_train_items': excluded_train_items})
        if response.status_code == 200:
            recommendations = response.json()['recommendations']
            return html.Ul([html.Li(f"{movie['title']} (Score: {movie['score']:.2f})") for movie in recommendations])
        else:
            return f'Error: Unable to get recommendations. Status code: {response.status_code}'
    else:  # movie_to_user
        if 'Selected Movie:' not in selected_movie:
            return 'Select a movie to get recommendations.'
        
        movie_title = selected_movie.replace('Selected Movie: ', '')
        if movie_title not in movie_titles:
            return 'Movie title not found in dataset.'

        movie_id = data_handler.movies['movieId'][data_handler.movies['title'] == movie_title].values[0]
        excluded_train_users = train_data.edge_index[0, train_data.edge_index[1, :] == movies_id_map[movie_id]].tolist()
        response = requests.post('http://localhost:5000/movie_to_user', json={'movie_id': movie_id, 'excluded_train_users': excluded_train_users})
        if response.status_code == 200:
            top_users = response.json()['top_users']
            return html.Ul([html.Li(f"User ID: {user['user_id']} (Score: {user['score']:.2f})") for user in top_users])
        else:
            return f'Error: Unable to get top users. Status code: {response.status_code}'

@app.callback(
    Output('visualization-graph', 'figure'),
    Input('visualization-type', 'value')
)

def update_visualization(visualization_type):
    if visualization_type == 'user-item-graph':
        response = requests.get('http://localhost:5000/get_embeddings')
        if response.status_code == 200:
            embeddings = response.json()
            user_embedding = torch.tensor(embeddings['user_embedding'])
            item_embedding = torch.tensor(embeddings['item_embedding'])
            
            G = create_user_item_graph(user_embedding, item_embedding)
            return plot_user_item_graph(G)
        else:
            return go.Figure()
    
    elif visualization_type == 'tsne-embedding':
        response = requests.get('http://localhost:5000/get_embeddings')
        if response.status_code == 200:
            embeddings = response.json()
            combined_embedding = torch.tensor(embeddings['user_embedding'] + embeddings['item_embedding'])
            labels = ['User' if i < len(embeddings['user_embedding']) else 'Item' for i in range(len(combined_embedding))]
            
            return plot_embedding_tsne(combined_embedding, labels)
        else:
            return go.Figure()

if __name__ == '__main__':
    # Load model and data handler
    data_handler = MovieLensDataHandler(os.path.join(DATA_DIR, "ratings.csv"), os.path.join(DATA_DIR, "movies.csv"))
    num_users, num_items = data_handler.get_num_users_items()

    model = LightGCN(num_users, num_items)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    train_data, _, _ = data_handler.get_datasets()
    train_data = train_data.to(device)

    app.run_server(debug=True, port=8050)
