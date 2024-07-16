import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Convolutional transformations for queries, keys, and values
        self.conv_query = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, groups=num_heads)
        self.conv_key = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, groups=num_heads)
        self.conv_value = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, groups=num_heads)

        # Output linear transformation
        self.fc_out = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x, y, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(2)
        height, width = x.size(3), x.size(4)

        # Linear transformation and splitting into multiple heads
        query = self.conv_query(x).view(batch_size, self.num_heads, self.head_dim, seq_len, height, width).transpose(2, 3)  # (batch_size, num_heads, seq_len, head_dim, height, width)
        key = self.conv_key(y).view(batch_size, self.num_heads, self.head_dim, seq_len, height, width).transpose(2, 3)  # (batch_size, num_heads, seq_len, head_dim, height, width)
        value = self.conv_value(y).view(batch_size, self.num_heads, self.head_dim, seq_len, height, width).transpose(2, 3)  # (batch_size, num_heads, seq_len, head_dim, height, width)

        # Attention scores computation
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len, height, width)

        if mask is not None:
            scores.masked_fill_(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len, height, width)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim, height, width)

        # Concatenate and linear transformation
        out = out.transpose(2, 3).contiguous().view(batch_size, seq_len, -1, height, width)  # (batch_size, seq_len, num_heads * head_dim, height, width)
        out = self.fc_out(out.transpose(1, 2))  # (batch_size, num_heads * head_dim, seq_len, height, width)
        out = out + x

        return out, attention_weights, value