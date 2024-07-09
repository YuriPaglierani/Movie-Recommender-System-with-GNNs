from typing import Dict, List, Union, Any
import plotly.graph_objs as go
import networkx as nx
import numpy as np
import plotly.express as px
import torch
from sklearn.manifold import TSNE
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from umap import UMAP


# for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

def create_user_item_graph(user_embedding, item_embedding, num_users=100, num_items=100):
    G = nx.Graph()
    
    # Add users and items as nodes
    for i in range(num_users):
        G.add_node(f'U{i}', bipartite=0)
    for i in range(num_items):
        G.add_node(f'I{i}', bipartite=1)
    
    # Add edges based on similarity
    for i in range(num_users):
        user_emb = user_embedding[i]
        similarities = torch.matmul(user_emb, item_embedding.t())
        top_items = torch.topk(similarities, 5).indices.tolist()
        for item in top_items:
            G.add_edge(f'U{i}', f'I{item}')
    
    return G

def plot_user_item_graph(G):
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='User-Item Interaction Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig

def analyze_user_recommendations(model: torch.nn.Module, user_id: int, data_handler: Any, n_neighbors: int = 15, min_dist: float = 0.1) -> px.scatter:
    """
    Analyze and visualize user recommendations using UMAP.

    This function generates a UMAP plot that visualizes the relationships between
    the main user, similar users, dissimilar users, and recommended movies in a
    2D embedding space.

    Args:
        model (torch.nn.Module): The trained recommendation model.
        user_id (int): The ID of the main user to analyze.
        data_handler (Any): The data handler object containing user and movie information.
        n_neighbors (int): The number of neighbors to consider in UMAP. Default is 15.
        min_dist (float): The minimum distance between points in UMAP. Default is 0.1.

    Returns:
        px.scatter: A Plotly Express scatter plot figure.
    """
    user_id_map = data_handler.user_id_map
    movie_id_map = data_handler.movie_id_map
    movies = data_handler.movies
    user_index = user_id_map[user_id]

    with torch.no_grad():
        all_user_embeddings, all_item_embeddings = model.get_embeddings(
            user_indices=torch.arange(len(user_id_map)),
            item_indices=torch.arange(len(movie_id_map))
        )

        all_user_embeddings = all_user_embeddings / torch.norm(all_user_embeddings, dim=1, keepdim=True)
        all_item_embeddings = all_item_embeddings / torch.norm(all_item_embeddings, dim=1, keepdim=True)

        main_user_embedding = all_user_embeddings[user_index].unsqueeze(0)
        
        user_scores = torch.matmul(main_user_embedding, all_user_embeddings.t()).squeeze()
        
        n_top_bottom = min(25, len(user_id_map) // 2 - 1)
        n_movies = min(50, len(movie_id_map))

        top_users = torch.topk(user_scores, k=n_top_bottom + 1, largest=True)
        bottom_users = torch.topk(user_scores, k=n_top_bottom, largest=False)
        
        if top_users.indices[0] == user_index:
            top_user_indices = top_users.indices[1:]
        else:
            top_user_indices = top_users.indices[:n_top_bottom]

        movie_scores = torch.matmul(main_user_embedding, all_item_embeddings.t()).squeeze()
        top_movies = torch.topk(movie_scores, k=n_movies, largest=True)

        embeddings_for_umap = torch.cat([
            main_user_embedding,
            all_user_embeddings[top_user_indices],
            all_user_embeddings[bottom_users.indices],
            all_item_embeddings[top_movies.indices]
        ], dim=0)
        
        umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        embeddings_2d = umap_reducer.fit_transform(embeddings_for_umap.cpu().numpy())

        plot_data = []
        
        plot_data.append({
            'x': embeddings_2d[0, 0],
            'y': embeddings_2d[0, 1],
            'type': 'Main User',
            'id': f"User {user_id}",
            'score': 1.0
        })
        
        for i, idx in enumerate(top_user_indices):
            plot_data.append({
                'x': embeddings_2d[i+1, 0],
                'y': embeddings_2d[i+1, 1],
                'type': 'Similar User',
                'id': f"User {list(user_id_map.keys())[list(user_id_map.values()).index(idx.item())]}",
                'score': user_scores[idx].item()
            })
        
        for i, idx in enumerate(bottom_users.indices):
            plot_data.append({
                'x': embeddings_2d[i+n_top_bottom+1, 0],
                'y': embeddings_2d[i+n_top_bottom+1, 1],
                'type': 'Dissimilar User',
                'id': f"User {list(user_id_map.keys())[list(user_id_map.values()).index(idx.item())]}",
                'score': user_scores[idx].item()
            })
        
        for i, idx in enumerate(top_movies.indices):
            movie_id = list(movie_id_map.keys())[list(movie_id_map.values()).index(idx.item()+len(user_id_map))]
            movie_title = movies[movies['movieId'] == movie_id].iloc[0]['title']
            plot_data.append({
                'x': embeddings_2d[i+2*n_top_bottom+1, 0],
                'y': embeddings_2d[i+2*n_top_bottom+1, 1],
                'type': 'Movie',
                'id': movie_title,
                'score': movie_scores[idx].item()
            })

        df = pd.DataFrame(plot_data)

        fig = px.scatter(df, x='x', y='y', color='type', symbol='type',
                         hover_name='id', hover_data=['score'],
                         color_discrete_map={
                             'Main User': '#FF0000',
                             'Similar User': '#00BFFF',
                             'Dissimilar User': '#FFA500',
                             'Movie': '#32CD32'
                         },
                         symbol_map={
                             'Main User': 'star',
                             'Similar User': 'circle',
                             'Dissimilar User': 'circle',
                             'Movie': 'diamond'
                         },
                         title=f'User-Movie Embedding Space (UMAP) for User {user_id}')

        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>Score: %{customdata[0]:.4f}<extra></extra>",
            marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
        )

        fig.update_layout(
            legend_title_text='Type',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            template='plotly_white',
            height=700,
            width=1000,
            font=dict(family="Arial", size=12),
            legend=dict(itemsizing='constant', title_font_size=14),
            title=dict(font=dict(size=18))
        )

        return fig
    
# def plot_embedding_tsne(embedding, labels):
#     tsne = TSNE(n_components=2, random_state=42)
#     embedding_2d = tsne.fit_transform(embedding.detach().cpu().numpy())
    
#     trace = go.Scatter(
#         x=embedding_2d[:, 0],
#         y=embedding_2d[:, 1],
#         mode='markers',
#         marker=dict(
#             size=8,
#             color=labels,
#             colorscale='Viridis',
#             showscale=True
#         ),
#         text=labels
#     )
    
#     layout = go.Layout(
#         title='t-SNE Visualization of Embeddings',
#         xaxis=dict(title='t-SNE 1'),
#         yaxis=dict(title='t-SNE 2')
#     )
    
#     fig = go.Figure(data=[trace], layout=layout)
#     return fig

def plot_histories():
    # Load data
    hist_train_loss = np.load('data/histories/hist_train_loss.npy')
    hist_val_loss = np.load('data/histories/hist_val_loss.npy')
    hist_val_recall = np.load('data/histories/hist_val_recall.npy')

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Training and Validation Loss", ""),
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=np.arange(len(hist_train_loss)), y=hist_train_loss, 
                            mode='lines', name='Train Loss', line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(hist_val_loss)), y=hist_val_loss, 
                            mode='lines', name='Validation Loss', line=dict(color='#ff7f0e', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(hist_val_recall)), y=hist_val_recall, 
                            mode='lines', name='Recall@k', line=dict(color='#2ca02c', width=2)), row=2, col=1)

    best_epoch = np.argmax(hist_val_recall)
    fig.add_annotation(x=best_epoch, y=hist_val_recall[best_epoch],
                    text=f"Best Model<br>Epoch {best_epoch}<br>Recall@k: {hist_val_recall[best_epoch]:.4f}",
                    bgcolor="#f0f0f0", bordercolor="#636363", borderwidth=2,
                    font=dict(size=12, color="#636363"), align="left",
                    row=2, col=1)

    fig.update_layout(
        height=800, width=1000,
        title_text="Model Training Results",
        title_font=dict(size=24),
        showlegend=True,
        legend=dict(x=1.05, y=0.5),
        plot_bgcolor='white',
        font=dict(family="Arial", size=14),
    )

    fig.update_xaxes(title_text="Epoch", showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
    fig.update_xaxes(title_text="Epoch", showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
    fig.update_yaxes(title_text="Loss", showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(title_text="Recall@k", showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)

    fig.show()

def plot_recommendations(recommendations: List[Dict[str, Union[str, float]]], user_id: int):
    """
    Create a plotly bar chart of the top 10 recommended movies.

    Args:
        recommendations (List[Dict[str, Union[str, float]]]): List of recommended movies with titles and scores.
    """
    
    titles = [rec['title'] for rec in recommendations]
    scores = [rec['score'] for rec in recommendations]

    fig = go.Figure(data=[go.Bar(x=scores, y=titles, orientation='h')])
    fig.update_layout(
        title=f'Top 10 Movie Recommendations for User {user_id}',
        xaxis_title='Recommendation Score',
        yaxis_title='Movie Title',
        yaxis=dict(autorange="reversed"),
        height=600,
        width=1000
    )
    fig.show()