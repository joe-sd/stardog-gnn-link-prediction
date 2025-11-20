"""
Graph Neural Network model for link prediction.
Uses GraphSAGE for node embeddings and a simple MLP for link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GNNEncoder(nn.Module):
    """
    GraphSAGE encoder that generates node embeddings.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings
            num_layers: Number of GraphSAGE layers
        """
        super(GNNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer: hidden_dim -> output_dim
        if num_layers > 1:
            self.layers.append(SAGEConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index):
        """
        Forward pass through the encoder.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x


class LinkPredictor(nn.Module):
    """
    Link prediction head that takes two node embeddings and predicts edge probability.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """
        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layer in MLP
        """
        super(LinkPredictor, self).__init__()
        
        # Simple MLP: concatenate embeddings -> hidden -> output
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability
        )
    
    def forward(self, src_emb, dst_emb):
        """
        Predict edge probability between source and destination nodes.
        
        Args:
            src_emb: Source node embeddings [num_pairs, embedding_dim]
            dst_emb: Destination node embeddings [num_pairs, embedding_dim]
        
        Returns:
            Edge probabilities [num_pairs, 1]
        """
        # Concatenate source and destination embeddings
        pair_emb = torch.cat([src_emb, dst_emb], dim=1)
        return self.mlp(pair_emb)


class LinkPredictionModel(nn.Module):
    """
    Complete model combining GNN encoder and link predictor.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dim: int = 64, num_layers: int = 2):
        """
        Args:
            input_dim: Dimension of input node features
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GNN layers
        """
        super(LinkPredictionModel, self).__init__()
        
        self.encoder = GNNEncoder(input_dim, hidden_dim, embedding_dim, num_layers)
        self.predictor = LinkPredictor(embedding_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_label_index):
        """
        Forward pass for link prediction.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph structure [2, num_edges]
            edge_label_index: Edge pairs to predict [2, num_pairs]
        
        Returns:
            Edge probabilities [num_pairs]
        """
        # Generate node embeddings
        node_emb = self.encoder(x, edge_index)
        
        # Get embeddings for source and destination nodes
        src_emb = node_emb[edge_label_index[0]]
        dst_emb = node_emb[edge_label_index[1]]
        
        # Predict edge probabilities
        edge_probs = self.predictor(src_emb, dst_emb).squeeze()
        
        return edge_probs

