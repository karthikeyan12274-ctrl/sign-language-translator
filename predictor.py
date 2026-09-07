import os
import torch
import numpy as np
import yaml

from models.ctc_decoder import SignRecognitionModel
from models.translation_transformer import TranslationTransformer
from preprocessing.feature_engineering import normalize_landmarks

class SignLanguagePredictor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gloss_vocab = self.config['data']['gloss_vocab']
        self.text_vocab = self.config['data']['text_vocab']
        
        self.gloss_to_idx = {gloss: idx for idx, gloss in enumerate(self.gloss_vocab)}
        self.text_to_idx = {word: idx for idx, word in enumerate(self.text_vocab)}
        
        self.recognition_model = None
        self.translation_model = None
        self.is_loaded = False
        
        self.load_models()

    def load_models(self):
        checkpoint_dir = self.config['data']['checkpoint_dir']
        rec_path = os.path.join(checkpoint_dir, "recognition_best.pth")
        trans_path = os.path.join(checkpoint_dir, "translation_best.pth")
        
        # 1. Initialize models
        num_glosses = len(self.gloss_vocab)
        input_dim = self.config['model']['spatial']['input_dim']
        hidden_dim = self.config['model']['spatial']['hidden_dim']
        rec_layers = self.config['model']['temporal']['num_layers']
        rec_heads = self.config['model']['temporal']['num_heads']
        rec_ff = self.config['model']['temporal']['feedforward_dim']
        rec_dropout = self.config['model']['temporal']['dropout']
        
        self.recognition_model = SignRecognitionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_glosses=num_glosses,
            num_layers=rec_layers,
            num_heads=rec_heads,
            feedforward_dim=rec_ff,
            dropout=rec_dropout
        ).to(self.device)
        
        src_vocab_size = len(self.gloss_vocab)
        tgt_vocab_size = len(self.text_vocab)
        trans_dim = self.config['model']['translation']['embed_dim']
        trans_layers = self.config['model']['translation']['num_layers']
        trans_heads = self.config['model']['translation']['num_heads']
        trans_ff = self.config['model']['translation']['feedforward_dim']
        trans_dropout = self.config['model']['translation']['dropout']
        
        self.translation_model = TranslationTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=trans_dim,
            num_layers=trans_layers,
            num_heads=trans_heads,
            feedforward_dim=trans_ff,
            dropout=trans_dropout
        ).to(self.device)
        
        # 2. Try to load weights
        if os.path.exists(rec_path) and os.path.exists(trans_path):
            try:
                rec_cp = torch.load(rec_path, map_location=self.device, weights_only=False)
                self.recognition_model.load_state_dict(rec_cp['model_state_dict'])
                
                trans_cp = torch.load(trans_path, map_location=self.device, weights_only=False)
                self.translation_model.load_state_dict(trans_cp['model_state_dict'])
                
                self.recognition_model.eval()
                self.translation_model.eval()
                self.is_loaded = True
                print("Models loaded successfully from checkpoints!")
            except Exception as e:
                print(f"Error loading checkpoints: {e}. Models initialized with random weights.")
        else:
            print("Checkpoints not found. Using untrained models with random weights (for demonstration).")

    def predict(self, landmarks_seq):
        """
        landmarks_seq: numpy array of shape (T, 225) representing raw keypoints
        Returns:
            glosses: list of strings (predicted glosses)
            translation: string (translated english sentence)
            confidence: float (model prediction confidence)
        """
        # Ensure correct shape
        if len(landmarks_seq.shape) != 2 or landmarks_seq.shape[1] != 225:
            raise ValueError(f"Landmarks sequence must be of shape (T, 225), got {landmarks_seq.shape}")
            
        T = landmarks_seq.shape[0]
        
        # If sequence is too short, return empty prediction
        if T < 5:
            return [], "Waiting for input...", 1.0

        # Normalization
        norm_seq = normalize_landmarks(landmarks_seq)
        
        # Convert to tensor and add batch dimension (1, T, 225)
        src_tensor = torch.tensor(norm_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.recognition_model.eval()
        self.translation_model.eval()
        
        with torch.no_grad():
            # 1. Sign Recognition (CTC Logits)
            logits = self.recognition_model(src_tensor) # (1, T, num_glosses)
            
            # Greedy Decode
            decoded_ids = self.recognition_model.decode_greedy(logits, [T])[0]
            
            # Map to gloss strings
            predicted_glosses = [self.gloss_vocab[idx] for idx in decoded_ids if idx < len(self.gloss_vocab)]
            
            # Confidence score estimation based on log probs
            probs = torch.exp(logits)
            max_probs, _ = torch.max(probs, dim=-1)
            mean_prob = max_probs.mean().item()
            
            # 2. Translation
            # If no glosses are detected, return empty translation
            if len(predicted_glosses) == 0:
                return [], "No signs detected.", mean_prob
                
            # Create input indices for translation: [SOS, ...gloss_ids..., EOS]
            gloss_indices = [self.gloss_to_idx['<SOS>']] + [self.gloss_to_idx[g] for g in predicted_glosses] + [self.gloss_to_idx['<EOS>']]
            gloss_tensor = torch.tensor([gloss_indices], dtype=torch.long, device=self.device)
            
            # Translate autoregressively
            translated_ids = self.translation_model.translate(
                src=gloss_tensor,
                src_vocab=self.gloss_vocab,
                tgt_vocab=self.text_vocab,
                device=self.device
            )
            
            # Convert indices to words, skip SOS, EOS, PAD
            words = []
            for idx in translated_ids:
                word = self.text_vocab[idx]
                if word not in ["<PAD>", "<SOS>", "<EOS>"]:
                    words.append(word)
                    
            translation_str = " ".join(words)
            
            return predicted_glosses, translation_str, mean_prob
            
    def get_status(self):
        return {
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "gloss_vocab_size": len(self.gloss_vocab),
            "text_vocab_size": len(self.text_vocab)
        }
