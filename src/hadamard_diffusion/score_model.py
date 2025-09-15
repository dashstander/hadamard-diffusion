import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from .layers import HadamardTransformerLayer, MatrixEmbedding, RMSNorm


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
        batch_size = x.shape[0]

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

    def get_score_fn(self, train=True, sampling=False):
        """Get a score function compatible with SEDD interface"""

        def score_fn(x, sigma):
            if train:
                self.train()
            else:
                self.eval()

            return self(x, sigma)

        return score_fn


class BinaryUniformGraph:
    """Simplified uniform graph for binary {-1, +1} states"""

    def __init__(self):
        self.dim = 2  # Two classes: -1 and +1
        self.absorb = False

    def value_to_index(self, x):
        """Convert {-1, +1} values to {0, 1} indices"""
        return torch.where(x == -1, 0, 1).long()

    def index_to_value(self, indices):
        """Convert {0, 1} indices to {-1, +1} values"""
        return torch.where(indices == 0, -1, 1)

    def sample_transition(self, x, sigma):
        """Sample from the uniform transition for binary case"""
        # Convert values to indices
        indices = self.value_to_index(x)

        # Uniform transition: stay with prob exp(-sigma), flip with prob 1-exp(-sigma)
        move_chance = 1 - torch.exp(-sigma)
        should_flip = torch.rand_like(x.float()) < move_chance

        # Flip indices where needed
        new_indices = torch.where(should_flip, 1 - indices, indices)

        # Convert back to values
        return self.index_to_value(new_indices)

    def score_entropy(self, log_score, sigma, x, x0):
        """Score entropy loss for uniform binary diffusion"""
        # Convert to indices
        x_indices = self.value_to_index(x)
        x0_indices = self.value_to_index(x0)

        # Extract scores for current states
        score = log_score.exp()  # Convert from log-space

        # For uniform graph: score_entropy = exp(score).mean() - score[x] + corrections
        batch_shape = x.shape[:-1] if x.dim() > 1 else x.shape

        # Positive term: E[exp(score)]
        pos_term = score.mean(dim=-1)

        # Negative term: score at current position
        neg_term = torch.gather(score, -1, x_indices.unsqueeze(-1)).squeeze(-1)

        # Constant correction term (simplified for uniform case)
        # This depends on whether we moved or stayed
        moved = (x_indices != x0_indices)
        esigma_minus_1 = torch.expm1(sigma.squeeze(-1))

        # Correction based on transition probabilities
        const_term = torch.where(
            moved,
            -torch.log(esigma_minus_1) / 2,  # If we moved
            torch.log1p(-torch.exp(-sigma.squeeze(-1))) / 2  # If we stayed
        )

        return pos_term - neg_term / 2 + const_term

    def sample_limit(self, *batch_dims):
        """Sample from limiting distribution (uniform over {-1, +1})"""
        indices = torch.randint(0, 2, batch_dims)
        return self.index_to_value(indices)


# Test script
if __name__ == "__main__":
    # Test the score model
    batch_size = 2
    matrix_size = 32

    # Create model
    model = HadamardScoreModel(matrix_size=matrix_size)
    graph = BinaryUniformGraph()

    # Create sample input
    x = graph.sample_limit(batch_size, matrix_size, matrix_size)
    sigma = torch.rand(batch_size, 1)  # Random noise levels

    print(f"Input shape: {x.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Input values: {x[0, :3, :3]}")  # Show a 3x3 corner

    # Forward pass
    log_scores = model(x, sigma)
    print(f"Output log scores shape: {log_scores.shape}")
    print(f"Score probabilities for first element: {log_scores[0, 0, 0].exp()}")

    # Test score entropy
    x_clean = graph.sample_limit(batch_size, matrix_size, matrix_size)
    x_noisy = graph.sample_transition(x_clean, sigma)

    score_entropy = graph.score_entropy(log_scores, sigma, x_noisy, x_clean)
    print(f"Score entropy shape: {score_entropy.shape}")
    print(f"Score entropy values: {score_entropy.mean()}")