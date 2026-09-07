import torch
import torch.nn as nn
import math
from .spatial_encoder import PositionalEncoding

class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, num_layers=4, num_heads=8, feedforward_dim=512, dropout=0.1):
        super().__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)
        self.embed_dim = embed_dim

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src shape: (B, src_seq_len)
        tgt shape: (B, tgt_seq_len)
        src_key_padding_mask shape: (B, src_seq_len) - True for pad tokens
        tgt_key_padding_mask shape: (B, tgt_seq_len) - True for pad tokens
        """
        # Embed and apply positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.embed_dim))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.embed_dim))
        
        # Look-ahead mask for decoder
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, src.device)
        
        # Transformer forward pass
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project output to target vocabulary size
        return self.fc_out(out)

    def translate(self, src, src_vocab, tgt_vocab, max_len=50, device="cpu"):
        """
        Autoregressive greedy decoding for translation inference.
        src shape: (1, src_len)
        """
        self.eval()
        with torch.no_grad():
            src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.embed_dim))
            memory = self.transformer.encoder(src_emb)
            
            # Start token
            sos_idx = tgt_vocab.index("<SOS>")
            eos_idx = tgt_vocab.index("<EOS>")
            
            ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(sos_idx)
            
            for i in range(max_len - 1):
                tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.embed_dim))
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1), device)
                
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                prob = self.fc_out(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                
                ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device).fill_(next_word)], dim=1)
                if next_word == eos_idx:
                    break
                    
            decoded_indices = ys.squeeze(0).tolist()
            return decoded_indices
