"""
Flame_Skimmer — Temporal GNN with Bayesian (MC Dropout) prediction head.

Spatial: GCN/GAT per timestep (same as Dusk_Crayfish).
Temporal: LSTM (same as Dusk_Crayfish).
Bayesian: MC Dropout — dropout layers stay active at inference time, and
predictions are sampled N times to produce a mean and standard deviation.

Use case: you want uncertainty estimates alongside predictions, so anomaly
scoring can ask "how far is this observation from the predicted distribution"
rather than just "how far from the point prediction." A spike that falls
within the model's high-uncertainty region is less anomalous than the same
spike where the model was confident — that's signal traditional models throw away.

Implementation note: this is "approximate" Bayesian, not full variational
inference. Each weight is still a point estimate; the dropout sampling acts
as a variational approximation to a Gaussian process. Cheap, well-supported
in the literature (Gal & Ghahramani 2016), and effective in practice.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch

from config.config import Config


class FlameSkimmer(nn.Module):
    """
    Temporal GNN with MC Dropout uncertainty estimation.

    forward(...) by default returns mean predictions (same shape as DuskCrayfish).
    Call forward(..., return_uncertainty=True) to also get the std-dev across
    Monte Carlo samples — this is what the anomaly detector should use.
    """

    # Number of forward passes used to estimate predictive distribution.
    # More = better uncertainty estimates but slower inference.
    # 30 is a reasonable default from the MC Dropout literature.
    MC_SAMPLES = 30

    def __init__(self, num_node_features):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.gnn_type = Config.GNN_TYPE
        gnn_layers = Config.GNN_LAYERS
        temporal_layers = Config.TEMPORAL_LAYERS

        # IMPORTANT: dropout rate matters more here than in DuskCrayfish because
        # it's how we draw posterior samples. Too low → uncertainty estimates
        # collapse. Too high → underfitting. 0.1-0.2 is the typical range.
        self.dropout_rate = max(Config.DROPOUT, 0.1)

        # ─── Spatial (GNN) ─────────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList()
        if self.gnn_type == "GCN":
            self.gnn_layers.append(GCNConv(num_node_features, self.hidden_dim))
            for _ in range(gnn_layers - 1):
                self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        elif self.gnn_type == "GAT":
            self.gnn_layers.append(
                GATConv(num_node_features, self.hidden_dim, heads=4, concat=True)
            )
            for _ in range(gnn_layers - 1):
                self.gnn_layers.append(
                    GATConv(self.hidden_dim * 4, self.hidden_dim)
                )

        lstm_input_dim = (
            self.hidden_dim * 4 if self.gnn_type == "GAT" else self.hidden_dim
        )

        # ─── Temporal (LSTM) ───────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=self.dropout_rate if temporal_layers > 1 else 0,
        )

        # ─── Output ────────────────────────────────────────────────────────
        self.output_layer = nn.Linear(self.hidden_dim, num_node_features)
        # Dropout is now a first-class layer, applied during BOTH training AND
        # inference. That's the key trick that makes this Bayesian.
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.ReLU()

    def _forward_single(self, x_sequence, edge_index, batch_size, num_nodes):
        """One forward pass with dropout active. Returns point predictions."""
        batch_size, seq_len, num_nodes, num_features = x_sequence.shape

        data_list = [
            Data(x=torch.zeros(num_nodes, num_features), edge_index=edge_index)
            for _ in range(batch_size)
        ]
        batch_loader = Batch.from_data_list(data_list).to(x_sequence.device)
        batch_edge_index = batch_loader.edge_index

        gnn_outputs = []
        for t in range(seq_len):
            x_t = x_sequence[:, t, :, :]
            x_flat = x_t.reshape(-1, num_features)
            h = x_flat
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, batch_edge_index)
                h = self.activation(h)
                h = self.dropout(h)  # active even in eval mode (see forward())
            h = h.reshape(batch_size, num_nodes, -1)
            h_graph = h.mean(dim=1)
            gnn_outputs.append(h_graph)

        temporal_input = torch.stack(gnn_outputs, dim=1)
        lstm_out, _ = self.lstm(temporal_input)
        last_hidden = lstm_out[:, -1, :]

        expanded = last_hidden.unsqueeze(1).expand(-1, num_nodes, -1)
        expanded = self.dropout(expanded)  # final dropout before output
        predictions = self.output_layer(expanded)

        return predictions

    def forward(self, x_sequence, edge_index, batch_size, num_nodes, return_uncertainty=False):
        """
        Default: returns point predictions (compatible with DuskCrayfish interface).
        return_uncertainty=True: returns (mean, std) tuple from MC sampling.

        During training, this is called normally and behaves like a regular model.
        During inference, the anomaly detector should call with return_uncertainty=True
        to get the full predictive distribution.
        """
        if not return_uncertainty:
            # Training path or simple inference — behave like DuskCrayfish.
            return self._forward_single(x_sequence, edge_index, batch_size, num_nodes)

        # MC Dropout: keep dropout active even though we're in eval mode.
        # nn.Dropout normally turns off when model.eval() is called; we override
        # it by explicitly calling .train() on just the dropout layers.
        was_training = self.training
        self.train()  # all dropout layers active
        # But we still want everything else (BatchNorm, etc.) in eval state if
        # they exist. This model has no BatchNorm, so .train() is fine.

        samples = []
        with torch.no_grad():
            for _ in range(self.MC_SAMPLES):
                pred = self._forward_single(x_sequence, edge_index, batch_size, num_nodes)
                samples.append(pred)

        if not was_training:
            self.eval()

        # Stack into (n_samples, batch, num_nodes, num_features)
        stacked = torch.stack(samples, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std