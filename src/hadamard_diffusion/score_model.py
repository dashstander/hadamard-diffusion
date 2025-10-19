import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax

from hadamard_diffusion.layers import (
    HadamardTransformerLayer,
    MatrixEmbedding,
    RMSNorm,
    HadamardRADDLayer
)


class HadamardScoreModel(nn.Module):
    """Score model for discrete diffusion on Hadamard matrices with {-1, +1} values"""

    def __init__(
        self,
        matrix_size=32,
        element_dim=128,
        pool_dim=64,
        num_heads=8,
        head_dim=16,
        ffn_hidden_dim=256,
        num_layers=6,
        time_dim=None
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.element_dim = element_dim
        self.num_classes = 2  # {-1, +1}
        self.time_dim = time_dim or element_dim

        # Input embedding: map {-1, +1} -> embeddings
        self.embedding = MatrixEmbedding(embed_dim=element_dim, size=matrix_size)

        # Stack of transformer layers
        self.layers = nn.ModuleList([
            HadamardTransformerLayer(
                element_dim=element_dim,
                pool_dim=pool_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                time_dim=self.time_dim
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(element_dim)

        # Output projection to score logits
        # Each element outputs 2 logits for {-1, +1} classes
        self.score_proj = nn.Linear(element_dim, self.num_classes)

    def forward(self, x, sigma):
        """
        Args:
            x: (batch, height, width) tensor with values in {-1, +1}
            sigma: (batch,) or (batch, 1) noise levels

        Returns:
            log_scores: (batch, height, width, num_classes) log probabilities
        """

        # Ensure sigma is properly shaped
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)

        # Convert to embeddings
        embedded = self.embedding(x)  # (batch, height, width, element_dim)

        # Process through transformer layers
        element_reps = embedded
        row_reps = None
        column_reps = None

        for layer in self.layers:
            element_reps, row_reps, column_reps = layer(
                element_reps, sigma.squeeze(-1), row_reps, column_reps
            )

        # Final normalization
        element_reps = self.final_norm(element_reps)

        # Project to score logits
        logits = self.score_proj(element_reps)  # (batch, height, width, 2)

        # Convert to log probabilities
        log_scores = log_softmax(logits, dim=-1)

        return log_scores


class HadamardRADDModel(nn.Module):
    """RADD (time-independent) version of Hadamard score model"""

    def __init__(
        self,
        matrix_size=32,
        element_dim=128,
        pool_dim=64,
        num_heads=8,
        head_dim=16,
        ffn_hidden_dim=256,
        num_layers=6
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.element_dim = element_dim
        self.num_classes = 2  # {-1, +1}

        # Input embedding: map {-1, +1} -> embeddings
        self.embedding = MatrixEmbedding(embed_dim=element_dim, size=matrix_size)

        # Stack of time-independent transformer layers
        self.layers = nn.ModuleList([
            HadamardRADDLayer(
                element_dim=element_dim,
                pool_dim=pool_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                ffn_hidden_dim=ffn_hidden_dim
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(element_dim)

        # Output projection to conditional probabilities
        # Each element outputs 2 probabilities for {-1, +1} classes
        self.prob_proj = nn.Linear(element_dim, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, height, width) tensor with values in {-1, +1}

        Returns:
            log_probs: (batch, height, width, num_classes) log conditional probabilities
        """
        # Convert to embeddings
        embedded = self.embedding(x)  # (batch, height, width, element_dim)

        # Process through transformer layers
        element_reps = embedded
        row_reps = None
        column_reps = None

        for layer in self.layers:
            element_reps, row_reps, column_reps = layer(element_reps, row_reps, column_reps)

        # Final normalization
        element_reps = self.final_norm(element_reps)

        # Project to conditional probability logits
        logits = self.prob_proj(element_reps)  # (batch, height, width, 2)

        # Convert to log probabilities using softmax
        log_probs = log_softmax(logits, dim=-1)

        return log_probs

    def probabilities(self, x):
        """
        Get conditional probabilities (non-log version)

        Args:
            x: (batch, height, width) tensor with values in {-1, +1}

        Returns:
            probs: (batch, height, width, num_classes) conditional probabilities
        """
        return softmax(self.forward(x), dim=-1)
