import plotly.graph_objs as go
import networkx as nx
from sklearn.manifold import TSNE

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