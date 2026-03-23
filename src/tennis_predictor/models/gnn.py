"""Graph Neural Network for intransitive player dominance.

Based on Clegg (2025): "Intransitive Player Dominance in Tennis using GNNs"
Key insight: Bookmakers are weakest on matches with high intransitive complexity
(A beats B, B beats C, but C beats A). A GNN on the player match graph can
capture these patterns that traditional rating systems miss.

Architecture:
- Players are nodes in a directed graph
- Edges represent recent match outcomes (winner -> loser)
- Node features: Elo, Glicko-2, rank, recent form, surface stats
- GNN layers aggregate neighborhood info (who you beat, who beat you)
- Final prediction combines GNN embeddings for both players

Requires: pip install torch torch-geometric (optional dependency)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, GATConv

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


if HAS_TORCH_GEOMETRIC:

    class TennisGNN(nn.Module):
        """GNN for learning player embeddings from the match graph."""

        def __init__(
            self,
            node_feature_dim: int = 16,
            hidden_dim: int = 64,
            embedding_dim: int = 32,
            n_layers: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.n_layers = n_layers

            # GNN layers (GraphSAGE for scalability)
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(node_feature_dim, hidden_dim))
            for _ in range(n_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, embedding_dim))

            # Batch normalization
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)
            ])

            self.dropout = dropout

            # Match prediction head: takes concatenated embeddings of both players
            self.predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Compute player embeddings from the match graph."""
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x

        def predict(
            self,
            embeddings: torch.Tensor,
            p1_indices: torch.Tensor,
            p2_indices: torch.Tensor,
        ) -> torch.Tensor:
            """Predict match outcomes from player embeddings."""
            p1_emb = embeddings[p1_indices]
            p2_emb = embeddings[p2_indices]
            combined = torch.cat([p1_emb, p2_emb], dim=1)
            return self.predictor(combined).squeeze()

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            p1_indices: torch.Tensor,
            p2_indices: torch.Tensor,
        ) -> torch.Tensor:
            embeddings = self.encode(x, edge_index)
            return self.predict(embeddings, p1_indices, p2_indices)


    def build_match_graph(
        matches: pd.DataFrame,
        player_features: dict[str, np.ndarray],
        lookback_days: int = 365,
        cutoff_date: pd.Timestamp | None = None,
    ) -> tuple[Data, dict[str, int]]:
        """Build a PyTorch Geometric graph from recent match history.

        Args:
            matches: DataFrame with match results (winner_id, loser_id, tourney_date).
            player_features: Dict mapping player_id -> feature vector.
            lookback_days: Only include matches from this many days back.
            cutoff_date: Only include matches before this date.

        Returns:
            Tuple of (PyG Data object, player_id -> node_index mapping).
        """
        if cutoff_date is not None:
            matches = matches[matches["tourney_date"] <= cutoff_date]

        if lookback_days:
            min_date = matches["tourney_date"].max() - pd.Timedelta(days=lookback_days)
            matches = matches[matches["tourney_date"] >= min_date]

        # Build node index
        all_players = set(matches["winner_id"].unique()) | set(matches["loser_id"].unique())
        player_to_idx = {pid: i for i, pid in enumerate(sorted(all_players))}
        n_nodes = len(player_to_idx)

        # Build edges (directed: winner -> loser)
        src = matches["winner_id"].map(player_to_idx).values
        dst = matches["loser_id"].map(player_to_idx).values
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        # Build node features
        feature_dim = next(iter(player_features.values())).shape[0] if player_features else 16
        x = torch.zeros((n_nodes, feature_dim), dtype=torch.float)
        for pid, idx in player_to_idx.items():
            if pid in player_features:
                x[idx] = torch.tensor(player_features[pid], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        return data, player_to_idx


    class GNNPredictor:
        """Sklearn-compatible wrapper around the TennisGNN."""

        def __init__(
            self,
            hidden_dim: int = 64,
            embedding_dim: int = 32,
            n_layers: int = 3,
            lr: float = 0.001,
            epochs: int = 100,
            batch_size: int = 512,
        ):
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.n_layers = n_layers
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.model = None
            self.graph_data = None
            self.player_to_idx = None

        def fit(self, graph_data: Data, matches: pd.DataFrame, y: np.ndarray):
            """Train the GNN on match outcomes."""
            self.graph_data = graph_data
            node_dim = graph_data.x.shape[1]

            self.model = TennisGNN(
                node_feature_dim=node_dim,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                n_layers=self.n_layers,
            )

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.BCELoss()

            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(
                    graph_data.x,
                    graph_data.edge_index,
                    torch.tensor(matches["p1_idx"].values, dtype=torch.long),
                    torch.tensor(matches["p2_idx"].values, dtype=torch.long),
                )

                loss = criterion(predictions, torch.tensor(y, dtype=torch.float))
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    print(f"  GNN Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

            return self

        def predict_proba(self, matches: pd.DataFrame) -> np.ndarray:
            """Predict match outcomes."""
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(
                    self.graph_data.x,
                    self.graph_data.edge_index,
                    torch.tensor(matches["p1_idx"].values, dtype=torch.long),
                    torch.tensor(matches["p2_idx"].values, dtype=torch.long),
                )
            proba = predictions.numpy()
            return np.column_stack([1 - proba, proba])

else:
    # Fallback when torch-geometric is not installed
    class GNNPredictor:
        """Placeholder — install torch-geometric for GNN support."""
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise ImportError(
                "GNN requires PyTorch Geometric. Install with: "
                "pip install 'tennis-predictor[ml-advanced]'"
            )

        def predict_proba(self, *args, **kwargs):
            raise ImportError("GNN requires PyTorch Geometric.")
