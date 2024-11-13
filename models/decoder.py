import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.positional import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_embed_dim: int,
        img_embed_dim: int,
        ff_dim: int,
        context_size: int,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.embed_pos = PositionalEncoding(word_embed_dim, context_size)
        self.context_size = context_size
        # Must return tensor of the same shape
        self.masked_attn = MaskedAttentionBlock(embedding_dim=word_embed_dim)
        self.cross_attn = CrossAttentionBlock(word_embed_dim, img_embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(word_embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, word_embed_dim),
            nn.Dropout(dropout),
        )

        self.embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.projection = nn.Linear(word_embed_dim, vocab_size)

    def forward(self, tokens: torch.LongTensor, img_emb: torch.Tensor):
        # tokens: [batch_size, seq_len]
        word_emb = self.embedding(tokens)
        word_emb = self.embed_pos(word_emb)
        # [batch_size, seq_len, embedding_dim]

        word_emb = self.masked_attn(word_emb)
        # [batch_size, seq_len, embedding_dim]

        combined_embeddings = self.cross_attn(word_emb, img_emb)

        # [seq_len, vocab_size]
        projected = self.projection(combined_embeddings)

        return projected


class MaskedAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
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
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, word_emb: torch.Tensor):
        # Embeddings: [seq_len, embedding_dim]
        Q = self.M_q(word_emb)

        # [seq_len, embedding_dim]
        K = self.M_k(word_emb)

        # [seq_len, seq_len]
        attn = (Q @ K.transpose(-1, -2)) / self.scaling_fac

        # seq_len = attn.shape[1]

        mask = torch.full_like(attn, float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        masked_attn = attn + mask

        # [batch_size, seq_len, seq_len]
        A = F.softmax(masked_attn, dim=-1)

        # [batch_size, seq_len, emb_dim]
        V = self.M_v(word_emb)

        # [batch_size, seq_len, embedding_dim]
        attn_emb = A @ V

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + word_emb)

        # [batch_size, seq_len, embedding_dim]
        return attn_emb


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        word_emb_dim: int,
        img_emb_dim: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.word_emb_dim = word_emb_dim
        self.img_emb_dim = img_emb_dim
        # self.combined_dim = min(word_emb_dim, img_emb_dim)
        self.scaling_fac = self.word_emb_dim ** (1 / 2)
        self.add_norm = add_norm
        self.M_q = nn.Linear(word_emb_dim, self.word_emb_dim)
        self.M_k = nn.Linear(img_emb_dim, self.word_emb_dim)
        self.M_v = nn.Linear(img_emb_dim, self.word_emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(word_emb_dim)

    def forward(self, word_emb: torch.Tensor, img_emb: torch.Tensor):
        Q = self.M_q(word_emb)
        # [batch_size, seq_len, word_emb_dim]

        K = self.M_k(img_emb)
        # [batch_size, seq_len, img_emb_dim]

        A = (Q @ K.transpose(-1, -2)) / self.scaling_fac
        # [batch_size, seq_len, seq_len]

        A = F.softmax(A, dim=-1)
        # [batch_size, seq_len, seq_len]

        V = self.M_v(img_emb)
        # [batch_size, seq_len, word_emb_dim]

        attn_emb = A @ V
        # [batch_size, seq_len, embedding_dim]

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + word_emb)

        # [batch_size, seq_len, embedding_dim]
        return attn_emb


if __name__ == "__main__":
    pass
