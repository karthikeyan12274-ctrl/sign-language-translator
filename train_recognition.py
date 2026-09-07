import os
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np

from training.dataset import SignLanguageDataset, collate_fn
from training.losses import RecognitionCTCLoss
from training.evaluate import calculate_wer
from models.ctc_decoder import SignRecognitionModel

def train_recognition():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and Loaders
    train_dataset = SignLanguageDataset("data/processed/train", config_path)
    val_dataset = SignLanguageDataset("data/processed/val", config_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['recognition']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['recognition']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize Model
    num_glosses = len(config['data']['gloss_vocab'])
    input_dim = config['model']['spatial']['input_dim']
    hidden_dim = config['model']['spatial']['hidden_dim']
    num_layers = config['model']['temporal']['num_layers']
    num_heads = config['model']['temporal']['num_heads']
    feedforward_dim = config['model']['temporal']['feedforward_dim']
    dropout = config['model']['temporal']['dropout']

    model = SignRecognitionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_glosses=num_glosses,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout
    ).to(device)

    # Loss and Optimizer
    criterion = RecognitionCTCLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['recognition']['learning_rate']
    )
    
    # Save directory
    os.makedirs(config['data']['checkpoint_dir'], exist_ok=True)
    best_val_loss = float('inf')

    # Training Loop
    epochs = config['training']['recognition']['epochs']
    print(f"Starting Recognition Model training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move coordinates to device
            landmarks = batch['landmarks'].to(device) # (B, T, input_dim)
            landmark_lens = batch['landmark_lens'].to(device)
            landmark_mask = batch['landmark_mask'].to(device)
            
            # targets: pad sequence to single 1D tensor for CTCLoss, or targets matrix.
            # PyTorch's CTCLoss allows target sequences to be 2D tensor of (B, max_target_length)
            # provided we pass target_lengths
            gloss_indices = batch['gloss_indices'].to(device) # (B, gloss_len)
            
            # For CTCLoss, targets should not contain SOS or EOS.
            # Let's strip the SOS (index 1) and EOS (index 2) from gloss_indices, or we can train including them.
            # In dataset.py, we added SOS (index 1) and EOS (index 2). Let's strip them for standard CTC gloss recognition.
            # Strip SOS and EOS
            clean_targets = []
            clean_target_lens = []
            for b in range(gloss_indices.size(0)):
                seq = batch['gloss_raw'][b] # original raw gloss tokens
                indices = [train_dataset.gloss_to_idx[g] for g in seq if g in train_dataset.gloss_to_idx]
                clean_targets.extend(indices)
                clean_target_lens.append(len(indices))
                
            clean_targets = torch.tensor(clean_targets, dtype=torch.long, device=device)
            clean_target_lens = torch.tensor(clean_target_lens, dtype=torch.long, device=device)

            optimizer.zero_grad()
            
            # Forward pass: outputs log-probabilities (B, T, num_glosses)
            log_probs = model(landmarks, src_key_padding_mask=landmark_mask)
            
            # Calculate CTCLoss
            loss = criterion(log_probs, clean_targets, landmark_lens, clean_target_lens)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        all_refs = []
        all_hyps = []
        
        with torch.no_grad():
            for batch in val_loader:
                landmarks = batch['landmarks'].to(device)
                landmark_lens = batch['landmark_lens'].to(device)
                landmark_mask = batch['landmark_mask'].to(device)
                gloss_indices = batch['gloss_indices'].to(device)
                
                clean_targets = []
                clean_target_lens = []
                for b in range(gloss_indices.size(0)):
                    seq = batch['gloss_raw'][b]
                    indices = [val_dataset.gloss_to_idx[g] for g in seq if g in val_dataset.gloss_to_idx]
                    clean_targets.extend(indices)
                    clean_target_lens.append(len(indices))
                    
                clean_targets = torch.tensor(clean_targets, dtype=torch.long, device=device)
                clean_target_lens = torch.tensor(clean_target_lens, dtype=torch.long, device=device)
                
                log_probs = model(landmarks, src_key_padding_mask=landmark_mask)
                loss = criterion(log_probs, clean_targets, landmark_lens, clean_target_lens)
                val_loss += loss.item()
                
                # Greedy decoding for validation predictions
                decoded_ids = model.decode_greedy(log_probs, landmark_lens.tolist())
                
                # Format references and hypotheses
                for b in range(len(decoded_ids)):
                    ref_glosses = batch['gloss_raw'][b]
                    hyp_glosses = [val_dataset.gloss_vocab[idx] for idx in decoded_ids[b] if idx < len(val_dataset.gloss_vocab)]
                    all_refs.append(ref_glosses)
                    all_hyps.append(hyp_glosses)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        wer = calculate_wer(all_refs, all_hyps)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WER: {wer:.4f}")
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['data']['checkpoint_dir'], "recognition_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'wer': wer
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path}")

    print("Recognition Model training completed!")

if __name__ == "__main__":
    train_recognition()
