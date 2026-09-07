import torch
import torch.nn as nn
from .spatial_encoder import SpatialEncoder
from .temporal_transformer import TemporalTransformer

class SignRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_glosses, num_layers=4, num_heads=8, feedforward_dim=512, dropout=0.1):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(input_dim, hidden_dim, dropout)
        self.temporal_transformer = TemporalTransformer(
            hidden_dim, num_layers, num_heads, feedforward_dim, dropout
        )
        # Output layer for CTC (gloss classifications including the CTC blank token at index 0)
        self.fc = nn.Linear(hidden_dim, num_glosses)

    def forward(self, x, src_key_padding_mask=None):
        """
        x shape: (B, T, input_dim)
        src_key_padding_mask shape: (B, T)
        Returns:
            logits: (B, T, num_glosses) - log softmax probabilities for CTCLoss
        """
        # 1. Spatial Encoding
        x = self.spatial_encoder(x)
        
        # 2. Temporal Transformer
        x = self.temporal_transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 3. Logits
        logits = self.fc(x)
        
        # CTC loss in PyTorch expects log-probabilities
        return torch.log_softmax(logits, dim=-1)

    def decode_greedy(self, logits, input_lengths):
        """
        Greedy decoding of CTC outputs.
        logits shape: (B, T, num_classes)
        input_lengths: list of frame lengths for each batch item
        Returns:
            decoded_sequences: List of lists containing predicted class IDs (excluding blank and duplicates)
        """
        B, T, C = logits.shape
        predictions = torch.argmax(logits, dim=-1) # (B, T)
        
        decoded_sequences = []
        for b in range(B):
            seq = []
            prev_val = -1
            length = input_lengths[b]
            for t in range(length):
                val = predictions[b, t].item()
                # 0 is the CTC blank token
                if val != 0 and val != prev_val:
                    seq.append(val)
                prev_val = val
            decoded_sequences.append(seq)
            
        return decoded_sequences
