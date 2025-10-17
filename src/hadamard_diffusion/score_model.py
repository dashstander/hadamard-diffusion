import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax

from .layers import HadamardTransformerLayer, MatrixEmbedding, RMSNorm, AttentionPooling, CrossAttention, MatrixElementAttention, MatrixColumnAttention, GeGLU


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
        # Convert to embeddings
        embedded = self.embedding(x)

        # Process through transformer layers
        element_reps = embedded
        row_reps = None
        column_reps = None

        for layer in self.layers:
            element_reps, row_reps, column_reps = layer(element_reps, row_reps, column_reps)

        # Final normalization
        element_reps = self.final_norm(element_reps)

        # Project to conditional probability logits
        logits = self.prob_proj(element_reps)

        # Convert to probabilities using softmax
        probs = softmax(logits, dim=-1)

        return probs


class HadamardRADDLayer(nn.Module):
    """Time-independent Hadamard Transformer layer"""

    def __init__(self, element_dim, pool_dim, num_heads, head_dim, ffn_hidden_dim):
        super().__init__()
        self.element_dim = element_dim
        self.pool_dim = pool_dim

        # Shared attention modules (for weight sharing between row/column)
        self.element_attention = MatrixElementAttention(element_dim, num_heads, head_dim)
        self.pool_attention = MatrixColumnAttention(pool_dim, num_heads, head_dim)

        # Pooling modules
        self.column_pooling = AttentionPooling(element_dim, pool_dim)
        self.row_pooling = AttentionPooling(element_dim, pool_dim)

        # Cross attention modules (shared weights)
        self.cross_attention = CrossAttention(element_dim, pool_dim, num_heads, head_dim)

        # Simple RMS normalization layers (no time conditioning)
        self.element_norm1 = RMSNorm(element_dim)
        self.element_norm2 = RMSNorm(element_dim)
        self.element_norm3 = RMSNorm(element_dim)
        self.pool_norm1 = RMSNorm(pool_dim)
        self.pool_norm2 = RMSNorm(pool_dim)

        # Feed-forward layers
        self.element_ffn = GeGLU(element_dim, ffn_hidden_dim)
        self.column_ffn = GeGLU(pool_dim, ffn_hidden_dim)
        self.row_ffn = GeGLU(pool_dim, ffn_hidden_dim)

    def forward(self, element_reps, row_reps=None, column_reps=None):
        """
        Args:
            element_reps: (batch, x_dim, y_dim, element_dim)
            row_reps: (batch, y_dim, pool_dim) or None
            column_reps: (batch, x_dim, pool_dim) or None

        Returns:
            tuple: (element_reps, row_reps, column_reps)
        """
        # 1. Element attention on rows (transpose to get row attention)
        element_reps = element_reps + self.element_attention(
            self.element_norm1(element_reps).permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        # 2. Element attention on columns
        element_reps = element_reps + self.element_attention(
            self.element_norm2(element_reps)
        )

        # 3. Pool to column and row representations
        if column_reps is None:
            column_reps = self.column_pooling(element_reps)
        if row_reps is None:
            # For row pooling: transpose to (batch, y_dim, x_dim, element_dim),
            # pool over x_dim to get (batch, y_dim, pool_dim)
            row_reps = self.row_pooling(element_reps.permute(0, 2, 1, 3))

        # 4. Column attention
        column_reps = column_reps + self.pool_attention(self.pool_norm1(column_reps))

        # 5. Row attention (reuse same weights)
        row_reps = row_reps + self.pool_attention(self.pool_norm2(row_reps))

        # 6. Cross attention: column_reps -> elements + row_reps -> elements
        cross_from_columns = self.cross_attention(element_reps, column_reps)
        cross_from_rows = self.cross_attention(element_reps, row_reps)
        element_reps = element_reps + cross_from_columns + cross_from_rows

        # 7. Feed-forward on all three representations (no time conditioning)
        element_reps = element_reps + self.element_ffn(self.element_norm3(element_reps))
        column_reps = column_reps + self.column_ffn(column_reps)
        row_reps = row_reps + self.row_ffn(row_reps)

        return element_reps, row_reps, column_reps


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

        # Ensure move_chance broadcasts correctly with x
        while move_chance.dim() < x.dim():
            move_chance = move_chance.unsqueeze(-1)

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
        sigma_flat = sigma.squeeze(-1)
        esigma_minus_1 = torch.expm1(sigma_flat)

        # Ensure proper broadcasting for batch dimensions
        while esigma_minus_1.dim() < moved.dim():
            esigma_minus_1 = esigma_minus_1.unsqueeze(-1)
        while sigma_flat.dim() < moved.dim():
            sigma_flat = sigma_flat.unsqueeze(-1)

        # Correction based on transition probabilities
        const_term = torch.where(
            moved,
            -torch.log(esigma_minus_1) / 2,  # If we moved
            torch.log1p(-torch.exp(-sigma_flat)) / 2  # If we stayed
        )

        return pos_term - neg_term / 2 + const_term

    def sample_limit(self, *batch_dims):
        """Sample from limiting distribution (uniform over {-1, +1})"""
        indices = torch.randint(0, 2, batch_dims)
        return self.index_to_value(indices)


# Test script
if __name__ == "__main__":
    # Test both models
    batch_size = 2
    matrix_size = 32

    # Create models
    score_model = HadamardScoreModel(matrix_size=matrix_size)
    radd_model = HadamardRADDModel(matrix_size=matrix_size)
    graph = BinaryUniformGraph()

    # Create sample input
    x = graph.sample_limit(batch_size, matrix_size, matrix_size)
    sigma = torch.rand(batch_size, 1)  # Random noise levels

    print(f"Input shape: {x.shape}")
    print(f"Input values: {x[0, :3, :3]}")  # Show a 3x3 corner

    # Test score model (time-dependent)
    print("\n=== Testing HadamardScoreModel (time-dependent) ===")
    log_scores = score_model(x, sigma)
    print(f"Output log scores shape: {log_scores.shape}")
    print(f"Score probabilities for first element: {log_scores[0, 0, 0].exp()}")

    # Test score entropy
    x_clean = graph.sample_limit(batch_size, matrix_size, matrix_size)
    x_noisy = graph.sample_transition(x_clean, sigma)
    score_entropy = graph.score_entropy(log_scores, sigma, x_noisy, x_clean)
    print(f"Score entropy shape: {score_entropy.shape}")
    print(f"Score entropy values: {score_entropy.mean()}")

    # Test RADD model (time-independent)
    print("\n=== Testing HadamardRADDModel (time-independent) ===")
    log_probs = radd_model(x)
    print(f"Output log probabilities shape: {log_probs.shape}")
    print(f"Conditional probabilities for first element: {log_probs[0, 0, 0].exp()}")

    # Test probabilities method
    probs = radd_model.probabilities(x)
    print(f"Direct probabilities shape: {probs.shape}")
    print(f"Direct probabilities for first element: {probs[0, 0, 0]}")

    # Parameter comparison
    score_params = sum(p.numel() for p in score_model.parameters())
    radd_params = sum(p.numel() for p in radd_model.parameters())
    print(f"\nParameter counts:")
    print(f"  Score model: {score_params:,}")
    print(f"  RADD model: {radd_params:,}")
    print(f"  Difference: {score_params - radd_params:,} (RADD should have fewer)")