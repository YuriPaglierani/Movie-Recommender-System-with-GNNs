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

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Movie Recommender System', className='text-center mb-4'),
            dbc.Card([
                dbc.CardBody([
                    dbc.RadioItems(
                        id='mode-selection',
                        options=[
                            {'label': 'User to Movie', 'value': 'user_to_movie'},
                            {'label': 'Movie to User', 'value': 'movie_to_user'}
                        ],
                        value='user_to_movie',
                        inline=True,
                        className='mb-3'
                    ),
                    html.Div(id='input-container', className='mb-3'),
                    dbc.Button('Get Recommendations', id='recommend-button', color='primary', className='mb-3'),
                    dcc.Loading(
                        id="loading-recommendations",
                        type="default",
                        children=html.Div(id='recommendations-output')
                    )
                ])
            ], className='mb-4'),
            dbc.Card([
                dbc.CardBody([
                    html.H2('Recommender System Visualizations', className='mb-3'),
                    dbc.Select(
                        id='visualization-type',
                        options=[
                            {'label': 'User-Item Interaction Graph', 'value': 'user-item-graph'},
                            {'label': 't-SNE Embedding Visualization', 'value': 'tsne-embedding'}
                        ],
                        value='user-item-graph',
                        className='mb-3'
                    ),
                    dcc.Loading(
                        id="loading-graph",
                        type="default",
                        children=dcc.Graph(id='visualization-graph')
                    )
                ])
            ])
        ], width=12)
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
], fluid=True)

@app.callback(
    Output('input-container', 'children'),
    Input('mode-selection', 'value')
)
def update_input(mode):
    if mode == 'user_to_movie':
        return dbc.Input(id='id-input', type='number', placeholder='Enter User ID')
    else:
        return html.Div([
            dbc.Button('Select a Movie', id='open-modal-button'),
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
    State('selected-movie', 'children'),
    prevent_initial_call=True
)

def update_recommendations(n_clicks, mode, id_value, selected_movie):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    try:
        if mode == 'user_to_movie':
            if id_value is None:
                return 'Please enter a User ID.'
            
            excluded_train_items = train_data.edge_index[1, train_data.edge_index[0, :] == int(id_value)].tolist()
            response = requests.post('http://localhost:5000/user_to_movie', 
                                     json={'user_id': int(id_value), 'excluded_train_items': excluded_train_items},
                                     timeout=10)
        else:  # movie_to_user
            if not selected_movie or 'Selected Movie:' not in selected_movie:
                return 'Please select a movie.'
            
            movie_title = selected_movie.replace('Selected Movie: ', '')
            movie_id = data_handler.movies['movieId'][data_handler.movies['title'] == movie_title].values[0]
            excluded_train_users = train_data.edge_index[0, train_data.edge_index[1, :] == data_handler.movie_id_map[movie_id]].tolist()
            response = requests.post('http://localhost:5000/movie_to_user', 
                                     json={'movie_id': movie_id, 'excluded_train_users': excluded_train_users},
                                     timeout=10)

        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        data = response.json()

        if mode == 'user_to_movie':
            recommendations = data.get('recommendations', [])
            return html.Ul([html.Li(f"{movie['title']} (Score: {movie['score']:.2f})") for movie in recommendations])
        else:
            top_users = data.get('top_users', [])
            return html.Ul([html.Li(f"User ID: {user['user_id']} (Score: {user['score']:.2f})") for user in top_users])

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return f'Error: Unable to get recommendations. Please check if the API server is running.'
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f'An unexpected error occurred: {str(e)}'

@app.callback(
    Output('visualization-graph', 'figure'),
    Input('visualization-type', 'value')
)
def update_visualization(visualization_type):
    try:
        response = requests.get('http://localhost:5000/get_embeddings')
        response.raise_for_status()  # Raises an HTTPError for bad responses
        embeddings = response.json()
        
        if 'error' in embeddings:
            raise Exception(embeddings['error'])
        
        user_embedding = torch.tensor(embeddings['user_embedding'])
        item_embedding = torch.tensor(embeddings['item_embedding'])
        
        if visualization_type == 'user-item-graph':
            G = create_user_item_graph(user_embedding, item_embedding)
            return plot_user_item_graph(G)
        elif visualization_type == 'tsne-embedding':
            combined_embedding = torch.cat((user_embedding, item_embedding), dim=0)
            labels = ['User' if i < len(user_embedding) else 'Item' for i in range(len(combined_embedding))]
            return plot_embedding_tsne(combined_embedding, labels)
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return go.Figure().add_annotation(text=f"Error: Unable to get embeddings. Please check if the API server is running.", showarrow=False)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return go.Figure().add_annotation(text=f"An unexpected error occurred: {str(e)}", showarrow=False)

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