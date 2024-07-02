import dash
import plotly.graph_objs as go
import requests
import os
import sys
import pandas as pd
from dash import dcc, html
# from import Input, Output, State

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data.dataset_handler import get_movie_genres

DATA_DIR = "data/movielens-25m"
# Load the movies dataset
movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Movie Recommender System'),
    
    html.Div([
        html.H2('Existing User Recommendations'),
        dcc.Input(id='user-id-input', type='number', placeholder='Enter User ID'),
        html.Button('Get Recommendations', id='recommend-button'),
        html.Div(id='recommendations-output')
    ]),
    
    html.Div([
        html.H2('New User Recommendations'),
        dcc.Dropdown(
            id='genre-dropdown',
            options=[{'label': genre, 'value': genre} for genre in get_movie_genres(movies)],
            multi=True,
            placeholder='Select favorite genres'
        ),
        html.Button('Get Recommendations for New User', id='new-user-recommend-button'),
        html.Div(id='new-user-recommendations-output')
    ]),
    
    html.Div([
        html.H2('Movie Genre Distribution'),
        dcc.Graph(id='genre-distribution-chart')
    ])
])

@app.callback(
    Output('recommendations-output', 'children'),
    Input('recommend-button', 'n_clicks'),
    State('user-id-input', 'value')
)
def update_recommendations(n_clicks, user_id):
    if n_clicks is None or user_id is None:
        return 'Enter a user ID and click the button to get recommendations.'
    
    response = requests.post('http://localhost:5000/recommend', json={'user_id': user_id})
    
    if response.status_code == 200:
        recommendations = response.json()['recommendations']
        return html.Ul([html.Li(f"{movie['title']} ({movie['genres']})") for movie in recommendations])
    else:
        return 'Error: Unable to get recommendations.'

@app.callback(
    Output('new-user-recommendations-output', 'children'),
    Input('new-user-recommend-button', 'n_clicks'),
    State('genre-dropdown', 'value')
)
def update_new_user_recommendations(n_clicks, genres):
    if n_clicks is None or not genres:
        return 'Select favorite genres and click the button to get recommendations.'
    
    response = requests.post('http://localhost:5000/add_user', json={'genres': genres})
    
    if response.status_code == 200:
        recommendations = response.json()['recommendations']
        return html.Ul([html.Li(f"{movie['title']} ({movie['genres']})") for movie in recommendations])
    else:
        return 'Error: Unable to get recommendations.'

@app.callback(
    Output('genre-distribution-chart', 'figure'),
    Input('recommend-button', 'n_clicks')
)
def update_genre_distribution(n_clicks):
    genres = get_movie_genres(movies)
    genre_counts = {genre: 0 for genre in genres}
    
    for _, movie in movies.iterrows():
        for genre in movie['genres'].split('|'):
            if genre in genre_counts:
                genre_counts[genre] += 1
    
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'data': [go.Bar(x=[genre for genre, _ in sorted_genres],
                        y=[count for _, count in sorted_genres])],
        'layout': go.Layout(title='Movie Genre Distribution',
                            xaxis={'title': 'Genre'},
                            yaxis={'title': 'Count'})
    }

if __name__ == '__main__':
    app.run_server(debug=True)