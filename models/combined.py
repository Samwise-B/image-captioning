import torch
import sys
from pathlib import Path

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from models.encoder import Encoder
from models.decoder import Decoder


class DoubleTrouble(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        patch_size: int,
        word_embed_dim: int,
        img_embed_dim: int,
        ff_dim_decoder: int,
        context_size: int,
        num_patches: int,
        num_layers_encoder: int,
        num_heads_encoder: int,
        num_heads_decoder: int,
        ff_dim_encoder: int,
        dropout: int = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            img_embed_dim,
            patch_size,
            num_patches=num_patches,
            num_layers=num_layers_encoder,
            num_heads=num_heads_encoder,
            ff_dim=ff_dim_encoder,
        )
        self.decoder = Decoder(
            vocab_size,
            word_embed_dim,
            img_embed_dim,
            num_heads_decoder,
            ff_dim_decoder,
            context_size,
        )

    def forward(self, tokens, patches):
        img_emb = self.encoder(patches)
        label_prediction = self.decoder(tokens, img_emb)
        return label_prediction
