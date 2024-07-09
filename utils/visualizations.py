from typing import Dict, List, Union
import plotly.graph_objs as go
import networkx as nx
import numpy as np
import torch
from sklearn.manifold import TSNE
import numpy as np
from plotly.subplots import make_subplots

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

def plot_embedding_tsne(embedding, labels):
    tsne = TSNE(n_components=2, random_state=42)
    embedding_2d = tsne.fit_transform(embedding.detach().cpu().numpy())
    
    trace = go.Scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=labels,
            colorscale='Viridis',
            showscale=True
        ),
        text=labels
    )
    
    layout = go.Layout(
        title='t-SNE Visualization of Embeddings',
        xaxis=dict(title='t-SNE 1'),
        yaxis=dict(title='t-SNE 2')
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return fig

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

def plot_recommendations(recommendations: List[Dict[str, Union[str, float]]]):
    """
    Create a plotly bar chart of the top 10 recommended movies.

    Args:
        recommendations (List[Dict[str, Union[str, float]]]): List of recommended movies with titles and scores.
    """
    titles = [rec['title'] for rec in recommendations]
    scores = [rec['score'] for rec in recommendations]

    fig = go.Figure(data=[go.Bar(x=scores, y=titles, orientation='h')])
    fig.update_layout(
        title='Top 10 Movie Recommendations',
        xaxis_title='Recommendation Score',
        yaxis_title='Movie Title',
        height=600,
        width=1000
    )
    fig.show()