import os
import torch
from torch.utils.data import DataLoader
import yaml

from training.dataset import SignLanguageDataset, collate_fn
from training.losses import TranslationLoss
from training.evaluate import calculate_bleu
from models.translation_transformer import TranslationTransformer

def train_translation():
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
        batch_size=config['training']['translation']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['translation']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize Model
    src_vocab_size = len(config['data']['gloss_vocab'])
    tgt_vocab_size = len(config['data']['text_vocab'])
    embed_dim = config['model']['translation']['embed_dim']
    num_layers = config['model']['translation']['num_layers']
    num_heads = config['model']['translation']['num_heads']
    feedforward_dim = config['model']['translation']['feedforward_dim']
    dropout = config['model']['translation']['dropout']

    model = TranslationTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout
    ).to(device)

    # Loss and Optimizer
    pad_idx = train_dataset.text_to_idx["<PAD>"]
    criterion = TranslationLoss(pad_idx=pad_idx).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['translation']['learning_rate']
    )
    
    # Save directory
    os.makedirs(config['data']['checkpoint_dir'], exist_ok=True)
    best_val_loss = float('inf')

    # Training Loop
    epochs = config['training']['translation']['epochs']
    print(f"Starting Translation Model training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            gloss_indices = batch['gloss_indices'].to(device) # (B, gloss_len)
            text_indices = batch['text_indices'].to(device) # (B, text_len)
            
            gloss_mask = batch['gloss_mask'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            # For teacher forcing:
            # Inputs to decoder: all tokens except <EOS>
            tgt_input = text_indices[:, :-1]
            # Targets: all tokens except <SOS>
            tgt_target = text_indices[:, 1:]
            
            # Adjust padding mask for target input
            tgt_mask_input = text_mask[:, :-1]
            
            optimizer.zero_grad()
            
            # Forward pass: outputs (B, tgt_len - 1, tgt_vocab_size)
            logits = model(
                src=gloss_indices, 
                tgt=tgt_input, 
                src_key_padding_mask=gloss_mask, 
                tgt_key_padding_mask=tgt_mask_input
            )
            
            # Loss calculation
            loss = criterion(logits, tgt_target)
            
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
                gloss_indices = batch['gloss_indices'].to(device)
                text_indices = batch['text_indices'].to(device)
                gloss_mask = batch['gloss_mask'].to(device)
                text_mask = batch['text_mask'].to(device)
                
                tgt_input = text_indices[:, :-1]
                tgt_target = text_indices[:, 1:]
                tgt_mask_input = text_mask[:, :-1]
                
                logits = model(
                    src=gloss_indices, 
                    tgt=tgt_input, 
                    src_key_padding_mask=gloss_mask, 
                    tgt_key_padding_mask=tgt_mask_input
                )
                loss = criterion(logits, tgt_target)
                val_loss += loss.item()
                
                # Autoregressive generation for BLEU calculation (sentence by sentence)
                for b in range(gloss_indices.size(0)):
                    src_seq = gloss_indices[b:b+1] # Keep batch dimension as 1: (1, gloss_len)
                    
                    # Generate translation indices
                    decoded_ids = model.translate(
                        src=src_seq,
                        src_vocab=val_dataset.gloss_vocab,
                        tgt_vocab=val_dataset.text_vocab,
                        device=device
                    )
                    
                    # Map indices back to words
                    ref_text = batch['text_raw'][b].lower().split()
                    
                    # Filter out SOS, EOS, PAD
                    hyp_words = [val_dataset.text_vocab[idx] for idx in decoded_ids if val_dataset.text_vocab[idx] not in ["<PAD>", "<SOS>", "<EOS>"]]
                    
                    all_refs.append(ref_text)
                    all_hyps.append(hyp_words)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        bleu = calculate_bleu(all_refs, all_hyps)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {bleu:.4f}")
        
        # Save checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['data']['checkpoint_dir'], "translation_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'bleu': bleu
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path}")

    print("Translation Model training completed!")

if __name__ == "__main__":
    train_translation()
