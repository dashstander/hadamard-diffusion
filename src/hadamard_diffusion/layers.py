import torch
from torch.nn import Module, Linear, Embedding, Parameter, LayerNorm
from torch.nn.functional import gelu, pad, softmax, scaled_dot_product_attention, silu
import math



class MatrixEmbedding(Module):

    def __init__(self, embed_dim: int, size: int = 32):
        super().__init__()
        self.embeddings = Embedding(3, embedding_dim=embed_dim, padding_idx=2)
        self.size = size
    

    def forward(self, tensors):
        size = tensors.shape[-1]
        tensors = torch.where(tensors == 1, 0, 1).to(torch.long)
        if size < self.size:
            amount = (self.size - size) // 2
            padding = (amount, amount, amount, amount)
            tensors = pad(tensors, padding, 2)
        return self.embeddings(tensors)
    

class MatrixElementAttention(Module):
    """Encoder multi-head attention where each matrix element can attend to itself and the other elements in its **column**.
    Have elements attend to their **row** by passing in `matrix.permute(0, 2, 1, 3)`
    """

    def __init__(self, element_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.model_dim = element_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = Linear(element_dim, num_heads * head_dim)
        self.k_proj = Linear(element_dim, num_heads * head_dim)
        self.v_proj = Linear(element_dim, num_heads * head_dim)
        self.o_proj = Linear(num_heads * head_dim, element_dim)
    
    def forward(self, tensors):
        batch, x_dim, y_dim, element_dim = tensors.shape
        # Batch over all of the columns of the matrix (as well as the batch dimension)
        tensors = tensors.reshape(-1, y_dim, element_dim)
        queries = self.q_proj(tensors).reshape(-1, y_dim, self.num_heads, self.head_dim)
        keys = self.k_proj(tensors).reshape(-1, y_dim, self.num_heads, self.head_dim)
        values = self.v_proj(tensors).reshape(-1, y_dim, self.num_heads, self.head_dim)

        attention = scaled_dot_product_attention(queries, keys, values)
        output = self.o_proj(attention.reshape(-1, y_dim, self.num_heads * self.head_dim))
        return output.reshape(batch, x_dim, y_dim, element_dim)


class ColumnEncoder(Module):
    """The idea here is to get a single vector for each column in the matrix. 
    """

    def __init__(self, element_dim, column_dim, hidden_dim, column_size):
        super().__init__()
        self.model_dim = element_dim
        self.W_up = Parameter(data = torch.randn((column_size, element_dim, hidden_dim)), requires_grad=True)
        self.W_down = Linear(hidden_dim, column_dim, bias=False)
    
    def forward(self, tensors):
        column_vecs = torch.einsum('...bxyt,yth -> ...bxh', tensors, self.W_up)
        return self.W_down(gelu(column_vecs))
    

class MatrixColumnAttention(Module):
    """Encoder multi-head attention where each matrix element can attend to itself and the other elements in its **column**.
    Have elements attend to their **row** by passing in `matrix.permute(0, 2, 1, 3)`
    """

    def __init__(self, column_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.model_dim = column_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = Linear(column_dim, num_heads * head_dim)
        self.k_proj = Linear(column_dim, num_heads * head_dim)
        self.v_proj = Linear(column_dim, num_heads * head_dim)
        self.o_proj = Linear(num_heads * head_dim, column_dim)

    def forward(self, tensors):
        batch, matrix_size, column_dim = tensors.shape
        queries = self.q_proj(tensors).reshape(-1, matrix_size, self.num_heads, self.head_dim)
        keys = self.k_proj(tensors).reshape(-1, matrix_size, self.num_heads, self.head_dim)
        values = self.v_proj(tensors).reshape(-1, matrix_size, self.num_heads, self.head_dim)

        attention = scaled_dot_product_attention(queries, keys, values)
        output = self.o_proj(attention.reshape(-1, matrix_size, self.num_heads * self.head_dim))
        return output.reshape(batch, matrix_size, column_dim)


class RMSNorm(Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class GeGLU(Module):
    """Gated Linear Unit with GeLU activation"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gate_proj = Linear(input_dim, hidden_dim, bias=False)
        self.up_proj = Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        gate = gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class AttentionPooling(Module):
    """Learn to pool elements within a column/row using attention weights"""

    def __init__(self, element_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.query = Parameter(torch.randn(element_dim))  # learnable pooling query
        self.key_proj = Linear(element_dim, element_dim)
        self.value_proj = Linear(element_dim, output_dim)

    def forward(self, tensors):
        # tensors: (batch, x_dim, y_dim, element_dim)
        batch, x_dim, y_dim, element_dim = tensors.shape

        # Reshape to pool over y_dim (elements in each column)
        reshaped = tensors.reshape(batch * x_dim, y_dim, element_dim)

        # Compute attention weights
        keys = self.key_proj(reshaped)  # (batch*x_dim, y_dim, element_dim)
        query = self.query.unsqueeze(0)  # (1, element_dim)

        # Attention scores
        scores = torch.einsum('bne,qe->bn', keys, query).unsqueeze(-1)  # (batch*x_dim, y_dim, 1)
        weights = softmax(scores, dim=1)  # (batch*x_dim, y_dim, 1)

        # Weighted sum
        values = self.value_proj(reshaped)  # (batch*x_dim, y_dim, output_dim)
        pooled = torch.sum(weights * values, dim=1)  # (batch*x_dim, output_dim)

        return pooled.reshape(batch, x_dim, self.output_dim)


class CrossAttention(Module):
    """Cross attention from pooled representations back to elements"""

    def __init__(self, element_dim, pool_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = Linear(element_dim, num_heads * head_dim)  # elements as queries
        self.k_proj = Linear(pool_dim, num_heads * head_dim)    # pooled as keys
        self.v_proj = Linear(pool_dim, num_heads * head_dim)    # pooled as values
        self.o_proj = Linear(num_heads * head_dim, element_dim)

    def forward(self, elements, pooled_reps):
        # elements: (batch, x_dim, y_dim, element_dim)
        # pooled_reps: (batch, x_dim, pool_dim) for columns OR (batch, y_dim, pool_dim) for rows

        batch, x_dim, y_dim, element_dim = elements.shape

        if pooled_reps.shape[1] == x_dim:  # column pooling case
            # Each element attends to its column representation
            # Reshape elements: (batch * y_dim, x_dim, element_dim)
            elements_reshaped = elements.permute(0, 2, 1, 3).reshape(batch * y_dim, x_dim, element_dim)
            # Expand pooled_reps: (batch * y_dim, x_dim, pool_dim)
            pooled_expanded = pooled_reps.unsqueeze(1).expand(batch, y_dim, x_dim, -1).reshape(batch * y_dim, x_dim, -1)
        else:  # row pooling case (pooled_reps.shape[1] == y_dim)
            # Each element attends to its row representation
            # Reshape elements: (batch * x_dim, y_dim, element_dim)
            elements_reshaped = elements.reshape(batch * x_dim, y_dim, element_dim)
            # Expand pooled_reps: (batch * x_dim, y_dim, pool_dim)
            pooled_expanded = pooled_reps.unsqueeze(1).expand(batch, x_dim, y_dim, -1).reshape(batch * x_dim, y_dim, -1)

        # Compute attention
        queries = self.q_proj(elements_reshaped).reshape(-1, elements_reshaped.shape[1], self.num_heads, self.head_dim)
        keys = self.k_proj(pooled_expanded).reshape(-1, pooled_expanded.shape[1], self.num_heads, self.head_dim)
        values = self.v_proj(pooled_expanded).reshape(-1, pooled_expanded.shape[1], self.num_heads, self.head_dim)

        attention = scaled_dot_product_attention(queries, keys, values)
        output = self.o_proj(attention.reshape(-1, elements_reshaped.shape[1], self.num_heads * self.head_dim))

        # Reshape back to original element shape
        if pooled_reps.shape[1] == x_dim:  # column case
            output = output.reshape(batch, y_dim, x_dim, element_dim).permute(0, 2, 1, 3)
        else:  # row case
            output = output.reshape(batch, x_dim, y_dim, element_dim)

        return output


class TimeEmbedding(Module):
    """Sinusoidal time embeddings for diffusion timesteps"""

    def __init__(self, embed_dim, max_time=1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time = max_time

        # Pre-compute frequency components
        half_dim = embed_dim // 2
        freqs = torch.exp(-math.log(max_time) * torch.arange(half_dim) / half_dim)
        self.register_buffer('freqs', freqs)

        # Project to desired dimensions
        self.time_proj = Linear(embed_dim, embed_dim)

    def forward(self, timesteps):
        # timesteps: (batch_size,) or (batch_size, 1)
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        # Sinusoidal embeddings
        args = timesteps[:, None] * self.freqs[None, :]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd embed_dim
        if embeddings.shape[-1] < self.embed_dim:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)

        return self.time_proj(embeddings)


class AdaRMSNorm(Module):
    """Adaptive RMS normalization conditioned on time embeddings"""

    def __init__(self, dim, time_dim):
        super().__init__()
        self.eps = 1e-8
        self.scale_proj = Linear(time_dim, dim)
        self.shift_proj = Linear(time_dim, dim)

    def forward(self, x, time_emb):
        # x: (..., dim), time_emb: (batch, time_dim)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms

        # Broadcast time conditioning
        batch_size = time_emb.shape[0]
        scale = self.scale_proj(time_emb).view(batch_size, *([1] * (x.dim() - 2)), -1)
        shift = self.shift_proj(time_emb).view(batch_size, *([1] * (x.dim() - 2)), -1)

        return normalized * (1 + scale) + shift


class HadamardTransformerLayer(Module):
    """Complete Hadamard Transformer layer with 3-tuple residual stream and time conditioning"""

    def __init__(self, element_dim, pool_dim, num_heads, head_dim, ffn_hidden_dim, time_dim=None):
        super().__init__()
        self.element_dim = element_dim
        self.pool_dim = pool_dim
        self.time_dim = time_dim or element_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(self.time_dim)

        # Shared attention modules (for weight sharing between row/column)
        self.element_attention = MatrixElementAttention(element_dim, num_heads, head_dim)
        self.pool_attention = MatrixColumnAttention(pool_dim, num_heads, head_dim)

        # Pooling modules
        self.column_pooling = AttentionPooling(element_dim, pool_dim)
        self.row_pooling = AttentionPooling(element_dim, pool_dim)

        # Cross attention modules (shared weights)
        self.cross_attention = CrossAttention(element_dim, pool_dim, num_heads, head_dim)

        # Time-conditioned normalization layers
        self.element_norm1 = AdaRMSNorm(element_dim, self.time_dim)
        self.element_norm2 = AdaRMSNorm(element_dim, self.time_dim)
        self.element_norm3 = AdaRMSNorm(element_dim, self.time_dim)
        self.pool_norm1 = AdaRMSNorm(pool_dim, self.time_dim)
        self.pool_norm2 = AdaRMSNorm(pool_dim, self.time_dim)

        # Feed-forward layers
        self.element_ffn = GeGLU(element_dim, ffn_hidden_dim)
        self.column_ffn = GeGLU(pool_dim, ffn_hidden_dim)
        self.row_ffn = GeGLU(pool_dim, ffn_hidden_dim)

        # Time conditioning for FFN
        self.element_time_proj = Linear(self.time_dim, element_dim)
        self.column_time_proj = Linear(self.time_dim, pool_dim)
        self.row_time_proj = Linear(self.time_dim, pool_dim)

    def forward(self, element_reps, timesteps, row_reps=None, column_reps=None):
        """
        Args:
            element_reps: (batch, x_dim, y_dim, element_dim)
            timesteps: (batch,) or (batch, 1) - diffusion timesteps
            row_reps: (batch, y_dim, pool_dim) or None
            column_reps: (batch, x_dim, pool_dim) or None

        Returns:
            tuple: (element_reps, row_reps, column_reps)
        """
        # Get time embeddings
        time_emb = self.time_embedding(timesteps)

        # 1. Element attention on rows (transpose to get row attention)
        element_reps = element_reps + self.element_attention(
            self.element_norm1(element_reps, time_emb).permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

        # 2. Element attention on columns
        element_reps = element_reps + self.element_attention(
            self.element_norm2(element_reps, time_emb)
        )

        # 3. Pool to column and row representations
        if column_reps is None:
            column_reps = self.column_pooling(element_reps)
        if row_reps is None:
            # For row pooling: transpose to (batch, y_dim, x_dim, element_dim),
            # pool over x_dim to get (batch, y_dim, pool_dim)
            row_reps = self.row_pooling(element_reps.permute(0, 2, 1, 3))

        # 4. Column attention
        column_reps = column_reps + self.pool_attention(self.pool_norm1(column_reps, time_emb))

        # 5. Row attention (reuse same weights)
        row_reps = row_reps + self.pool_attention(self.pool_norm2(row_reps, time_emb))

        # 6. Cross attention: column_reps -> elements + row_reps -> elements
        cross_from_columns = self.cross_attention(element_reps, column_reps)
        cross_from_rows = self.cross_attention(element_reps, row_reps)
        element_reps = element_reps + cross_from_columns + cross_from_rows

        # 7. Feed-forward on all three representations with time conditioning
        element_time_cond = self.element_time_proj(time_emb)
        column_time_cond = self.column_time_proj(time_emb)
        row_time_cond = self.row_time_proj(time_emb)

        # Apply time conditioning to FFN inputs
        element_reps = element_reps + self.element_ffn(
            self.element_norm3(element_reps, time_emb) + element_time_cond.view(-1, 1, 1, self.element_dim)
        )
        column_reps = column_reps + self.column_ffn(
            column_reps + column_time_cond.view(-1, 1, self.pool_dim)
        )
        row_reps = row_reps + self.row_ffn(
            row_reps + row_time_cond.view(-1, 1, self.pool_dim)
        )

        return element_reps, row_reps, column_reps


