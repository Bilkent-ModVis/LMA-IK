"""LSTM Synthesizer that maps dense end-effector positions to full-body rotations.

See Section 3 of the paper. The architecture is a single-layer LSTM with
hidden dimension 128 followed by a dense projector with three 512-node
layers, ending in joint rotations expressed in the 6D representation of
Zhou et al. (2019).
"""

import torch
import torch.nn as nn


class Synthesizer(nn.Module):
    """LSTM + dense projector from dense end-effector positions to joint angles.

    The model takes a sequence of dense end-effector positions and a vector
    of LMA-based style descriptors per sequence, and predicts full-body
    joint rotations in 6D representation.
    """

    def __init__(self,
                 angles_dim: int,
                 positions_dim: int,
                 conditions_dim: int = 4,
                 hidden_dim: int = 128,
                 fc_dim: int = 512,
                 num_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = positions_dim + conditions_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.num_layers = num_layers
        self.angles_dim = angles_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pose_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.fc_dim),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.LeakyReLU(),
            nn.Linear(self.fc_dim, self.angles_dim),
        )

    def forward(self,
                dense_trajectory: torch.Tensor,
                conditions: torch.Tensor) -> torch.Tensor:
        """Predict per-frame joint rotations in 6D representation.

        Args:
            dense_trajectory: End-effector positions of shape
                (B, T, positions_dim).
            conditions: Style descriptors of shape (B, conditions_dim) or
                (B, T, conditions_dim).

        Returns:
            Predicted joint rotations of shape (B, T, angles_dim).
        """
        batch_size, seq_len, _ = dense_trajectory.shape

        if conditions.dim() == 2:
            conditions_expanded = conditions.unsqueeze(1).expand(-1, seq_len, -1)
        elif conditions.dim() == 3:
            conditions_expanded = conditions
        else:
            raise ValueError(f"Invalid conditions shape: {tuple(conditions.shape)}")

        combined = torch.cat([dense_trajectory, conditions_expanded], dim=2)
        lstm_output, _ = self.lstm(combined)
        return self.pose_projector(lstm_output)

    def _init_weights(self, module: nn.Module) -> None:
        """Optional Kaiming / orthogonal init suited to LSTM + LeakyReLU."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.0)
