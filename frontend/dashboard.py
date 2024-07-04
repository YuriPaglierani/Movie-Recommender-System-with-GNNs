import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import requests
import os
import sys
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.visualizations import create_user_item_graph, plot_user_item_graph, plot_embedding_tsne

DATA_DIR = "data/movielens-25m"
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
app = dash.Dash(__name__)

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
    
    html.Div([
        dcc.Input(id='id-input', type='number', placeholder='Enter ID'),
        html.Button('Get Recommendations', id='recommend-button'),
        html.Div(id='recommendations-output')
    ]),
    
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
    ])
])

@app.callback(
    Output('recommendations-output', 'children'),
    Input('recommend-button', 'n_clicks'),
    State('mode-selection', 'value'),
    State('id-input', 'value')
)
def update_recommendations(n_clicks, mode, id_value):
    if n_clicks is None or id_value is None:
        return 'Enter an ID and click the button to get recommendations.'
    
    if mode == 'user_to_movie':
        response = requests.post('http://localhost:5000/user_to_movie', json={'user_id': id_value})
        if response.status_code == 200:
            recommendations = response.json()['recommendations']
            return html.Ul([html.Li(f"{movie['title']} (Score: {movie['score']:.2f})") for movie in recommendations])
        else:
            return 'Error: Unable to get recommendations.'
    else:  # movie_to_user
        response = requests.post('http://localhost:5000/movie_to_user', json={'movie_id': id_value})
        if response.status_code == 200:
            top_users = response.json()['top_users']
            return html.Ul([html.Li(f"User ID: {user['user_id']} (Score: {user['score']:.2f})") for user in top_users])
        else:
            return 'Error: Unable to get top users.'

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
    app.run_server(debug=True)