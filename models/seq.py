import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BERT(nn.Module):
    def __init__(
        self,
        magic_layer: nn.Module,
        vocab_size: int,
        embedding_dim: int,
        context_size: int,
    ):
        super().__init__()

        self.embed_pos = PositionalEncoding(embedding_dim, context_size)
        self.context_size = context_size
        # Must return tensor of the same shape
        self.magic_layer = magic_layer

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.LongTensor):
        # tokens: [seq_len]

        # [seq_len, embedding_dim]
        embeddings = self.embedding(tokens)
        pos_embeddings = self.embed_pos(embeddings)

        # [seq_len, embedding_dim]
        transformed = self.magic_layer(pos_embeddings)

        # [seq_len, vocab_size]
        projected = self.projection(transformed)

        return projected


class UnbatchedSingleHeadTransformer(nn.Module):
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

    def forward(self, embeddings: torch.Tensor):
        # Embeddings: [seq_len, embedding_dim]
        Q = self.M_q(embeddings)

        # [seq_len, embedding_dim]
        K = self.M_k(embeddings)

        # [seq_len, seq_len]
        attn = (Q @ K.transpose(-1, -2)) / self.scaling_fac

        # seq_len = attn.shape[1]

        mask = torch.full_like(attn, float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        masked_attn = attn + mask

        # [batch_size, seq_len, seq_len]
        A = F.softmax(masked_attn, dim=-1)

        # [batch_size, seq_len, emb_dim]
        V = self.M_v(embeddings)

        # [batch_size, seq_len, embedding_dim]
        attn_emb = A @ V

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + embeddings)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(attn_emb)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) for positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # Initialize positional encodings with zeros
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch compatibility
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class OwnSingleHeadTransformer(nn.Module):
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

    def forward(self, x):
        embeddings, mask = x
        # Embeddings: [batch_size, seq_len, embedding_dim]
        Q = self.M_q(embeddings)

        # [batch_size, seq_len, embedding_dim]
        K = self.M_k(embeddings)

        # [batch_size, seq_len, seq_len]
        A_prime = torch.bmm(Q, K.transpose(-1, -2)) / self.scaling_fac

        seq_len = A_prime.shape[1]

        attn_mask = mask.unsqueeze(-2).repeat(1, seq_len, 1)

        masked_attn = torch.where(attn_mask, A_prime, float("-inf"))

        # [batch_size, seq_len, seq_len]
        A = F.softmax(A_prime, dim=-1)

        # [batch_size, seq_len, emb_dim]
        V = self.M_v(embeddings)

        # [batch_size, seq_len, embedding_dim]
        attn_emb = torch.bmm(A, V)

        attn_emb = self.attn_dropout(attn_emb)

        if self.add_norm:
            attn_emb = self.norm(attn_emb + embeddings)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(attn_emb), mask


class OwnMultiHeadTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        add_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embedding_dim
        self.scaling_fac = self.embed_dim ** (1 / 2)
        self.add_norm = add_norm
        if embedding_dim % num_heads:
            raise Exception("Embed dim not divisible by num of heads")
        self.head_dim = embedding_dim // num_heads

        self.M_ks = nn.ModuleList(
            [nn.Linear(embedding_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.M_qs = nn.ModuleList(
            [nn.Linear(embedding_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.M_vs = nn.ModuleList(
            [nn.Linear(embedding_dim, self.head_dim) for _ in range(num_heads)]
        )

        self.concat_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        embeddings, mask = x
        # Embeddings: [batch_size, seq_len, embedding_dim]

        # [num_heads, batch_size, seq_len, head_dim]
        Qs = [M_q(embeddings) for M_q in self.M_qs]
        Ks = [M_k(embeddings) for M_k in self.M_ks]

        seq_len = embeddings.shape[1]

        attn_mask = mask.unsqueeze(-2).repeat(1, seq_len, 1)
        # [num_heads, batch_size, seq_len, seq_len]
        A_primes = [
            torch.bmm(Q, K.transpose(-1, -2)) / self.scaling_fac for Q, K in zip(Qs, Ks)
        ]

        masked_attns = [
            torch.where(attn_mask, A_prime, float("-inf")) for A_prime in A_primes
        ]

        # num_heads[batch_size, seq_len, seq_len]
        As = [F.softmax(A, dim=-1) for A in A_primes]

        Vs = [M_v(embeddings) for M_v in self.M_vs]

        # num_heads[batch_size, seq_len, head_dim]
        Hs = [torch.bmm(A, V) for A, V in zip(As, Vs)]

        # [batch_size, seq_len, num_heads*head_dim = embed_dim]
        H = torch.cat(Hs, dim=-1)

        H = self.concat_proj(H)
        H = self.attn_dropout(H)

        if self.add_norm:
            H = self.norm(H + embeddings)

        # [batch_size, seq_len, embedding_dim]
        return self.ff(H), mask


if __name__ == "__main__":
    pass
