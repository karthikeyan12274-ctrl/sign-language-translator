import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers=4, num_heads=8, feedforward_dim=512, dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        x shape: (B, T, hidden_dim)
        src_key_padding_mask shape: (B, T) -> Boolean mask (True means masked out/padded)
        """
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out
