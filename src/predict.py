"""
Run inference to predict missing links in the graph.
"""

import torch
import numpy as np
from build_pyg_graph import load_from_csv
from model import LinkPredictionModel


def predict_missing_links(model, data, top_k=10):
    """
    Predict top-K most likely missing edges.
    
    Args:
        model: Trained LinkPredictionModel
        data: PyTorch Geometric Data object
        top_k: Number of top predictions to return
    
    Returns:
        List of tuples: (source_uri, target_uri, probability)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    model.eval()
    
    # Get all possible node pairs (excluding existing edges)
    num_nodes = data.x.shape[0]
    existing_edges = set()
    
    # Store existing edges as (src, dst) tuples
    for i in range(data.edge_index.shape[1]):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # Generate all possible pairs
    all_pairs = []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst and (src, dst) not in existing_edges:
                all_pairs.append((src, dst))
    
    print(f"Evaluating {len(all_pairs)} possible edge pairs...")
    
    # Batch predictions to avoid memory issues
    batch_size = 1000
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(all_pairs), batch_size):
            batch_pairs = all_pairs[i:i+batch_size]
            edge_label_index = torch.tensor(batch_pairs, dtype=torch.long).t().to(device)
            
            probs = model(data.x, data.edge_index, edge_label_index)
            all_probs.extend(probs.cpu().numpy())
    
    # Get top-K predictions
    all_probs = np.array(all_probs)
    top_indices = np.argsort(all_probs)[-top_k:][::-1]
    
    # Convert to URIs and return
    predictions = []
    idx_to_uri = data.idx_to_uri
    
    for idx in top_indices:
        src_idx, dst_idx = all_pairs[idx]
        prob = all_probs[idx]
        predictions.append((
            idx_to_uri[src_idx],
            idx_to_uri[dst_idx],
            float(prob)
        ))
    
    return predictions


if __name__ == '__main__':
    # Load data from Stardog-extracted CSV files
    print("Loading graph data from Stardog...")
    data = load_from_csv('../data/nodes_from_stardog.csv', '../data/edges_from_stardog.csv')
    
    # Load trained model
    input_dim = data.x.shape[1]
    model = LinkPredictionModel(
        input_dim=input_dim,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2
    )
    
    model.load_state_dict(torch.load('../data/model.pt', map_location='cpu'))
    print("Model loaded from data/model.pt\n")
    
    # Predict top 10 missing links
    predictions = predict_missing_links(model, data, top_k=10)
    
    print("Top 10 predicted missing links:")
    print("-" * 80)
    for i, (src_uri, dst_uri, prob) in enumerate(predictions, 1):
        print(f"{i}. {src_uri}")
        print(f"   -> {dst_uri}")
        print(f"   Probability: {prob:.4f}\n")

