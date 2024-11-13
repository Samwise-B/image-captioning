import torch
from encoder import Encoder
from decoder import Decoder


class DoubleTrouble(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        patch_size: int,
        word_embed_dim: int,
        img_embed_dim: int,
        ff_dim_decoder: int,
        context_size: int,
        num_layers_encoder: int,
        ff_dim_encoder: int,
        dropout: int = 0.1,
    ):
        self.encoder = Encoder(
            img_embed_dim,
            patch_size,
            num_layers=num_layers_encoder,
            ff_dim=ff_dim_encoder,
        )
        self.decoder = Decoder(
            vocab_size, word_embed_dim, img_embed_dim, ff_dim_decoder, context_size
        )

    def forward(self, tokens, patches):
        img_emb = self.encoder(patches)
        label_prediction = self.decoder(tokens, img_emb)
        return label_prediction
