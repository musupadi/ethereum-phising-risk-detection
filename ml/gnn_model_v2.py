"""
ml/gnn_model_v2.py
------------------
GNN model — two architecture options:
  - GraphSAGE (default): better neighbor aggregation for sparse transaction graphs
  - GCN (legacy): kept for backward compatibility

Key design decisions vs original:
  - 2 layers instead of 4–6: prevents over-smoothing on graphs with ~8K nodes
  - SAGEConv aggregates own features + neighbor mean separately (robust to degree variance)
  - lr=0.01 (was 0.001): converges faster without quality loss
  - Best model weights are restored after early stopping
  - Real transaction edges supported via build_graph_with_real_edges()
  - kNN fallback (k=15) when real edges are unavailable
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class SAGEModelV2(nn.Module):
    """
    2-layer GraphSAGE.

    SAGEConv(u→v): h_v = W1*h_v + W2*mean(h_u for u in N(v))
    Keeps own-feature contribution explicit — better than GCN's symmetric
    normalisation when node degrees vary widely (as in transaction graphs).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class GCNModelV2(nn.Module):
    """Legacy GCN kept for backward compatibility."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.fc = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            if residual.shape == x.shape and layer_idx > 0:
                x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


class GraphPhiV2:
    """
    GNN wrapper — GraphSAGE by default, GCN as optional legacy.

    Graph construction priority:
      1. Real transaction edges (build_graph_with_real_edges) — most meaningful
      2. kNN feature-similarity fallback (build_graph_from_train_only)
    """

    def __init__(
        self,
        model_type: str = "sage",
        input_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._best_state: Optional[dict] = None

        if self.model_type == "sage":
            self.model = SAGEModelV2(
                input_dim, hidden_dim, dropout=0.5, num_layers=num_layers
            ).to(DEVICE)
        elif self.model_type == "gcn":
            self.model = GCNModelV2(
                input_dim, hidden_dim, dropout=0.3, num_layers=num_layers
            ).to(DEVICE)
        else:
            raise ValueError(f"model_type must be 'sage' or 'gcn', got '{model_type}'")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=5e-4,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

        log.info(
            "Initialized %s V2: hidden=%d layers=%d device=%s",
            model_type.upper(), hidden_dim, num_layers, DEVICE,
        )

    # ── Graph builders ─────────────────────────────────────────────────────────

    def build_graph_with_real_edges(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        edge_list: list[list[int]],
    ) -> Tuple[Data, np.ndarray, np.ndarray]:
        """
        Build a PyG graph using real Ethereum transaction edges.

        Node index convention (must match edge_list):
          0 .. n_train-1          → training nodes (in order of X_train rows)
          n_train .. n_total-1    → test nodes     (in order of X_test rows)
        """
        n_train = len(X_train)
        n_test = len(X_test)
        n_total = n_train + n_test

        x = torch.FloatTensor(np.vstack([X_train, X_test])).to(DEVICE)
        y = torch.LongTensor(np.hstack([y_train, y_test])).to(DEVICE)

        if edge_list:
            arr = np.unique(np.array(edge_list, dtype=np.int64), axis=0)
            edge_index = torch.LongTensor(arr.T).to(DEVICE)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(DEVICE)

        self_loops = torch.arange(n_total, dtype=torch.long).repeat(2, 1).to(DEVICE)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[n_train:] = True

        graph = Data(x=x, edge_index=edge_index, y=y)
        log.info(
            "Real-edge graph: %d nodes (%d train / %d test), %d transaction edges",
            n_total, n_train, n_test, graph.num_edges - n_total,
        )
        return graph, train_mask, test_mask

    def build_graph_from_train_only(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: list = None,
    ) -> Tuple[Data, np.ndarray, np.ndarray]:
        """
        kNN fallback: build a feature-similarity graph.

        k=15 (up from the previous k=5) gives denser connectivity so
        2-layer message passing can reach a meaningful neighborhood.
        Only training nodes are used to build edges; test nodes connect
        to their nearest training neighbours to avoid data leakage.
        """
        n_train = len(X_train)
        n_test = len(X_test)
        n_total = n_train + n_test

        x = torch.FloatTensor(np.vstack([X_train, X_test])).to(DEVICE)
        y = torch.LongTensor(np.hstack([y_train, y_test])).to(DEVICE)

        edges = []
        k_train = min(15 if n_train >= 5000 else 10, n_train - 1)

        if n_train > 1 and k_train > 0:
            nn_train = NearestNeighbors(n_neighbors=k_train + 1, metric="cosine")
            nn_train.fit(X_train)
            train_nbrs = nn_train.kneighbors(X_train, return_distance=False)
            for i, nbrs in enumerate(train_nbrs):
                for j in nbrs:
                    if j != i:
                        edges.append([i, int(j)])
                        edges.append([int(j), i])

        k_cross = min(5, n_train)
        if n_test > 0 and n_train > 0:
            nn_cross = NearestNeighbors(n_neighbors=k_cross, metric="cosine")
            nn_cross.fit(X_train)
            cross_nbrs = nn_cross.kneighbors(X_test, return_distance=False)
            for t, nbrs in enumerate(cross_nbrs):
                t_idx = n_train + t
                for j in nbrs:
                    edges.append([t_idx, int(j)])
                    edges.append([int(j), t_idx])

        if not edges:
            edges = [[i, i + 1] for i in range(min(n_train - 1, 1))]
            edges += [[i + 1, i] for i in range(min(n_train - 1, 1))]
        if not edges:
            edges = [[0, 0]]

        edge_index = torch.LongTensor(np.array(edges, dtype=np.int64).T).to(DEVICE)
        self_loops = torch.arange(n_total, dtype=torch.long).repeat(2, 1).to(DEVICE)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[n_train:] = True

        graph = Data(x=x, edge_index=edge_index, y=y)
        log.info(
            "kNN graph (k=%d): %d nodes, %d edges",
            k_train, n_total, graph.num_edges - n_total,
        )
        return graph, train_mask, test_mask

    # ── Training ───────────────────────────────────────────────────────────────

    def train_epoch(self, graph: Data, train_mask=None) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(graph.x, graph.edge_index)
        mask = train_mask if train_mask is not None else slice(None)
        loss = self.loss_fn(out[mask], graph.y[mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, graph: Data, mask=None) -> dict:
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            pred = out.argmax(dim=1)
            y_true = graph.y[mask].cpu().numpy() if mask is not None else graph.y.cpu().numpy()
            y_pred = pred[mask].cpu().numpy() if mask is not None else pred.cpu().numpy()
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
        test_mask=None,
        epochs: int = 200,
        early_stopping_patience: int = 25,
    ) -> dict:
        """
        Train with class-balanced loss and best-weight restoration.

        Early stopping monitors F1 on `test_mask`; when patience is exhausted the
        model is rolled back to the epoch with the highest F1 (not the last epoch).
        Pass a train-internal VALIDATION mask here, never the held-out test mask,
        otherwise the reported test score is optimistically biased.
        """
        if train_mask is not None:
            y_train = graph.y[train_mask].cpu().numpy()
            counts = np.bincount(y_train)
            n_samples = len(y_train)
            weights = torch.FloatTensor(
                [n_samples / (2.0 * c) for c in counts]
            ).to(DEVICE)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            log.info(
                "Class weights: normal=%.3f  phishing=%.3f",
                weights[0].item(), weights[1].item(),
            )

        best_f1 = 0.0
        patience_counter = 0
        self._best_state = None
        history: dict[str, list] = {"train_loss": [], "test_f1": []}

        for epoch in range(epochs):
            loss = self.train_epoch(graph, train_mask)
            history["train_loss"].append(loss)

            if test_mask is not None:
                metrics = self.evaluate(graph, test_mask)
                f1 = metrics["f1"]
                history["test_f1"].append(f1)

                if f1 > best_f1:
                    best_f1 = f1
                    patience_counter = 0
                    self._best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if (epoch + 1) % 20 == 0:
                    log.info(
                        "[Epoch %3d] loss=%.4f  test_f1=%.4f  best=%.4f  patience=%d/%d",
                        epoch + 1, loss, f1, best_f1,
                        patience_counter, early_stopping_patience,
                    )

                if patience_counter >= early_stopping_patience:
                    log.info(
                        "Early stopping at epoch %d (best F1=%.4f)",
                        epoch + 1, best_f1,
                    )
                    break
            else:
                if (epoch + 1) % 20 == 0:
                    log.info("[Epoch %3d] loss=%.4f", epoch + 1, loss)

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            log.info("Restored best weights (F1=%.4f)", best_f1)

        return history

    def predict(self, graph: Data, mask=None) -> Tuple[np.ndarray, np.ndarray]:
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
        torch.save(self.model.state_dict(), path)
        log.info("Model saved → %s", path)

    def load(self, path: Path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        log.info("Model loaded ← %s", path)
