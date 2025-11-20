"""
Training script for link prediction model.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from build_pyg_graph import load_from_csv
from model import LinkPredictionModel


def train(model, data, epochs=100, lr=0.01):
    """
    Train the link prediction model.
    
    Args:
        model: LinkPredictionModel instance
        data: PyTorch Geometric Data object
        epochs: Number of training epochs
        lr: Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Split edges into train/validation (simple split for demo)
    num_edges = data.edge_index.shape[1]
    train_size = int(0.8 * num_edges)
    
    train_edge_index = data.edge_index[:, :train_size]
    val_edge_index = data.edge_index[:, train_size:]
    
    print(f"Training on {train_size} edges, validating on {num_edges - train_size} edges")
    print(f"Using device: {device}\n")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Positive edges (existing edges in training set)
        pos_edge_index = train_edge_index
        
        # Negative edges (sampled non-edges)
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=data.x.shape[0],
            num_neg_samples=pos_edge_index.shape[1]
        )
        
        # Combine positive and negative edges
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([
            torch.ones(pos_edge_index.shape[1]),
            torch.zeros(neg_edge_index.shape[1])
        ]).to(device)
        
        # Forward pass
        edge_probs = model(data.x, train_edge_index, edge_label_index)
        
        # Compute loss
        loss = criterion(edge_probs, edge_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Validation positive edges
                val_pos_edge_index = val_edge_index
                val_neg_edge_index = negative_sampling(
                    edge_index=val_edge_index,
                    num_nodes=data.x.shape[0],
                    num_neg_samples=val_pos_edge_index.shape[1]
                )
                
                val_edge_label_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim=1)
                val_edge_labels = torch.cat([
                    torch.ones(val_pos_edge_index.shape[1]),
                    torch.zeros(val_neg_edge_index.shape[1])
                ]).to(device)
                
                val_probs = model(data.x, train_edge_index, val_edge_label_index)
                val_loss = criterion(val_probs, val_edge_labels)
                
                # Compute accuracy
                predictions = (val_probs > 0.5).float()
                accuracy = (predictions == val_edge_labels).float().mean()
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {loss.item():.4f}")
                print(f"  Val Loss: {val_loss.item():.4f}")
                print(f"  Val Accuracy: {accuracy.item():.4f}\n")
    
    print("Training complete!")
    return model


if __name__ == '__main__':
    # Load data from Stardog-extracted CSV files
    print("Loading graph data from Stardog...")
    data = load_from_csv('../data/nodes_from_stardog.csv', '../data/edges_from_stardog.csv')
    
    # Create model
    input_dim = data.x.shape[1]
    model = LinkPredictionModel(
        input_dim=input_dim,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    trained_model = train(model, data, epochs=100, lr=0.01)
    
    # Save model
    torch.save(trained_model.state_dict(), '../data/model.pt')
    print("\nModel saved to data/model.pt")

