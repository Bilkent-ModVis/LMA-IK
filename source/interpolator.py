"""Conditional VAE that upsamples sparse end-effector keyframes to dense paths.

See Section 3 of the paper. The model is trained on motion windows of fixed
length and conditioned on the initial and final end-effector positions and
the four LMA-inspired style descriptors (V, H, P, R).
"""

import torch
import torch.nn as nn


class _Encoder(nn.Module):
    """LSTM encoder mapping a dense end-effector sequence to (mu, log_var)."""

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, self.seq_len, -1)
        _, (hidden, _) = self.lstm(x_flat)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class _Decoder(nn.Module):
    """LSTM decoder reconstructing the sequence from (z, condition, style)."""

    def __init__(self,
                 seq_len: int,
                 condition_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc_combine = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self,
                z: torch.Tensor,
                condition: torch.Tensor,
                style: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        combined = torch.cat([
            z,
            condition.view(batch_size, -1),
            style.view(batch_size, -1),
        ], dim=1)
        lstm_input = self.fc_combine(combined).unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(lstm_input)
        output = self.fc_out(lstm_out)
        return output.view(batch_size, self.seq_len, -1, 3)


class Interpolator(nn.Module):
    """CVAE that maps sparse end-effector keyframes to dense paths.

    The encoder takes the full dense sequence and outputs latent
    parameters. The decoder is conditioned on the initial and final
    end-effector positions and the LMA-based style descriptors.
    """

    def __init__(self,
                 seq_len: int,
                 num_points: int,
                 num_coords: int,
                 encoder_hidden_dim: int,
                 decoder_hidden_dim: int,
                 latent_dim: int,
                 num_style_descriptors: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        input_dim = num_points * num_coords
        condition_dim = 2 * num_points * num_coords + num_style_descriptors

        self.encoder = _Encoder(seq_len, input_dim, encoder_hidden_dim, latent_dim)
        self.decoder = _Decoder(seq_len, condition_dim, latent_dim, decoder_hidden_dim, input_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, style: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        endpoints = x[:, [0, -1], :, :]
        condition = torch.cat([
            endpoints[:, 0].reshape(x.size(0), -1),
            endpoints[:, 1].reshape(x.size(0), -1),
        ], dim=1)

        reconstructed = self.decoder(z, condition, style)
        return reconstructed, mu, log_var

    def generate(self,
                 condition: torch.Tensor,
                 style: torch.Tensor,
                 device: str = 'cpu') -> torch.Tensor:
        """Sample a dense end-effector sequence given conditioning inputs."""
        self.eval()
        with torch.no_grad():
            batch_size = condition.size(0)
            z = torch.randn(batch_size, self.latent_dim, device=device)
            return self.decoder(z, condition, style)
