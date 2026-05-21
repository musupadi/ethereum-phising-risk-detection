"""
ml/gnn_model.py
---------------
Graph Neural Network model for phishing detection.
Uses PyTorch Geometric for efficient graph learning.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ── Settings ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class GCNModel(nn.Module):
    """Graph Convolutional Network for node classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass."""
        # Layer 1
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.gcn3(x, edge_index)
        x = F.relu(x)
        
        # Classification head
        x = self.fc(x)
        return x


class GAT_Model(nn.Module):
    """Graph Attention Network for node classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.5):
        """
        Initialize GAT model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass."""
        # Layer 1
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        
        # Classification head
        x = self.fc(x)
        return x


class GraphPhi:
    """High-level wrapper for GNN-based phishing detection."""

    def __init__(self, model_type: str = "gcn", input_dim: int = 18, hidden_dim: int = 64):
        """
        Initialize GNN wrapper.
        
        Args:
            model_type: "gcn" or "gat"
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
        """
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize model
        if self.model_type == "gcn":
            self.model = GCNModel(input_dim, hidden_dim).to(DEVICE)
        elif self.model_type == "gat":
            self.model = GAT_Model(input_dim, hidden_dim).to(DEVICE)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=5e-4,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        log.info("Initialized %s model on device: %s", model_type.upper(), DEVICE)

    def build_graph_from_df(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        label_col: str = "label",
        add_self_loops: bool = True,
    ) -> Data:
        """
        Build PyTorch Geometric Data object from dataframe.
        Assumes edges come from transaction relationships or proximity.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: Column names for features
            label_col: Column name for labels
            add_self_loops: Whether to add self-loops to edge_index
            
        Returns:
            torch_geometric.data.Data object
        """
        log.info("Building graph from %d nodes", len(df))
        
        # Node features
        x = torch.FloatTensor(df[feature_cols].fillna(0).values).to(DEVICE)
        
        # Labels
        y = torch.LongTensor(df[label_col].values).to(DEVICE)
        
        # Build edge_index from similarity or correlation
        # Simple approach: connect nodes with similar features
        num_nodes = len(df)
        num_neighbors = min(10, num_nodes // 2)  # Connect to k-nearest neighbors
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(x.cpu().numpy())
        
        # Build edges from top similarities (excluding self)
        edges = []
        for i in range(num_nodes):
            # Get top k neighbors (excluding self)
            neighbors = np.argsort(similarity[i])[::-1]
            neighbors = neighbors[neighbors != i][:num_neighbors]
            
            for j in neighbors:
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph
        
        if not edges:
            # Fallback: create simple chain if no edges
            edges = [[i, i + 1] for i in range(num_nodes - 1)]
            edges.extend([[i + 1, i] for i in range(num_nodes - 1)])
        
        edge_index = torch.LongTensor(np.array(edges, dtype=int).T).to(DEVICE)
        
        # Add self-loops
        if add_self_loops:
            self_loops = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1).to(DEVICE)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        # Create Data object
        graph = Data(x=x, edge_index=edge_index, y=y)
        log.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.num_edges)
        
        return graph

    def train_epoch(self, graph: Data, train_mask=None):
        """Train for one epoch."""
        self.model.train()
        
        self.optimizer.zero_grad()
        out = self.model(graph.x, graph.edge_index)
        
        if train_mask is not None:
            loss = self.loss_fn(out[train_mask], graph.y[train_mask])
        else:
            loss = self.loss_fn(out, graph.y)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, graph: Data, mask=None) -> dict:
        """Evaluate model on given mask."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            pred = out.argmax(dim=1)
            
            if mask is not None:
                y_true = graph.y[mask].cpu().numpy()
                y_pred = pred[mask].cpu().numpy()
            else:
                y_true = graph.y.cpu().numpy()
                y_pred = pred.cpu().numpy()
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    def fit(
        self,
        graph: Data,
        train_mask=None,
        val_mask=None,
        epochs: int = 100,
        early_stopping_patience: int = 20,
    ) -> dict:
        """
        Train model.
        
        Args:
            graph: PyTorch Geometric Data object
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes
            epochs: Number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history
        """
        log.info("Training %s for %d epochs", self.model_type.upper(), epochs)
        
        best_val_f1 = 0
        patience_counter = 0
        history = {"train_loss": [], "val_f1": []}
        
        for epoch in range(epochs):
            loss = self.train_epoch(graph, train_mask)
            history["train_loss"].append(loss)
            
            if val_mask is not None:
                val_metrics = self.evaluate(graph, val_mask)
                val_f1 = val_metrics["f1"]
                history["val_f1"].append(val_f1)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 20 == 0:
                    log.info(
                        "[Epoch %3d] Loss=%.4f  Val F1=%.4f  (patience: %d/%d)",
                        epoch + 1, loss, val_f1, patience_counter, early_stopping_patience,
                    )
                
                if patience_counter >= early_stopping_patience:
                    log.info("Early stopping at epoch %d", epoch + 1)
                    break
            else:
                if (epoch + 1) % 20 == 0:
                    log.info("[Epoch %3d] Loss=%.4f", epoch + 1, loss)
        
        return history

    def predict(self, graph: Data, mask=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels and confidence.
        
        Args:
            graph: PyTorch Geometric Data object
            mask: Optional mask for subset of nodes
            
        Returns:
            (predictions, confidence_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            proba = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            if mask is not None:
                pred = pred[mask]
                proba = proba[mask]
        
        return pred.cpu().numpy(), proba.cpu().numpy()

    def save(self, path: Path):
        """Save model to disk."""
        torch.save(self.model.state_dict(), path)
        log.info("Model saved → %s", path)

    def load(self, path: Path):
        """Load model from disk."""
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        log.info("Model loaded ← %s", path)
