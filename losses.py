import torch
import torch.nn as nn

class RecognitionCTCLoss(nn.Module):
    def __init__(self, zero_infinity=True):
        super().__init__()
        # CTC blank token is index 0
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: (B, T, num_classes) -> needs to be permuted to (T, B, num_classes) for PyTorch CTCLoss
        targets: (B, tgt_len) or flat targets
        input_lengths: (B,)
        target_lengths: (B,)
        """
        # Permute log_probs to shape (T, B, C)
        log_probs_perm = log_probs.transpose(0, 1)
        return self.ctc_loss(log_probs_perm, targets, input_lengths, target_lengths)

class TranslationLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, logits, targets):
        """
        logits shape: (B, T, vocab_size) -> needs to be reshaped to (B * T, vocab_size)
        targets shape: (B, T) -> needs to be reshaped to (B * T)
        """
        vocab_size = logits.size(-1)
        return self.criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

