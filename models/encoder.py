import torch
import torch.nn.functional as F
from torch import nn
from positional import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_layers: int,
        ff_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Linear(patch_size, embed_dim)
        num_patches = self.embedding.shape[1]
        self.embed_pos = PositionalEncoding(embed_dim, num_patches)
        self.attn_block = nn.ModuleList(
            [EncoderAttentionBlock(embed_dim, ff_dim) for _ in range(num_layers)]
        )

    def forward(self, tokens: torch.LongTensor):
        embeddings = self.embedding(tokens)
        embeddings = self.embed_pos(embeddings)
        # batch_size, seq_len, embedding_dim
        for layer in self.attn_block:
            embeddings = layer(embeddings)
        # batch_size, seq_len, embedding_dim

        return embeddings


class EncoderAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.scaling_fac = self.embed_dim ** (1 / 2)
        self.add_norm = add_norm
        self.M_q = nn.Linear(embedding_dim, embedding_dim)
        self.M_k = nn.Linear(embedding_dim, embedding_dim)
        self.M_v = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, word_emb: torch.Tensor):
        # Embeddings: [seq_len, embedding_dim]
        Q = self.M_q(word_emb)

        # [seq_len, embedding_dim]
        K = self.M_k(word_emb)

        # [seq_len, seq_len]
        attn = (Q @ K.transpose(-1, -2)) / self.scaling_fac

        # [batch_size, seq_len, seq_len]
        A = F.softmax(attn, dim=-1)

        # [batch_size, seq_len, emb_dim]
        V = self.M_v(word_emb)

        # [batch_size, seq_len, embedding_dim]
        attn_emb = A @ V

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + word_emb)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(attn_emb)
