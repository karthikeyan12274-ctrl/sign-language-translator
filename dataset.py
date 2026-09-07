import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from preprocessing.feature_engineering import normalize_landmarks

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.gloss_vocab = self.config['data']['gloss_vocab']
        self.text_vocab = self.config['data']['text_vocab']
        
        self.gloss_to_idx = {gloss: idx for idx, gloss in enumerate(self.gloss_vocab)}
        self.text_to_idx = {word: idx for idx, word in enumerate(self.text_vocab)}
        
        # Gather all .npz samples in the folder
        self.samples = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.samples) == 0:
            print(f"Warning: No data samples found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Landmarks: [T, 225]
        landmarks = data['landmarks']
        # Apply normalization
        landmarks = normalize_landmarks(landmarks)
        
        # Glosses: list of strings -> list of indices with <SOS> and <EOS>
        glosses = data['glosses'].tolist()
        gloss_indices = [self.gloss_to_idx['<SOS>']] + [self.gloss_to_idx[g] for g in glosses if g in self.gloss_to_idx] + [self.gloss_to_idx['<EOS>']]
        
        # Translation: string -> list of words -> list of indices
        translation_text = str(data['translation'])
        words = translation_text.lower().split()
        text_indices = [self.text_to_idx['<SOS>']] + [self.text_to_idx[w] for w in words if w in self.text_to_idx] + [self.text_to_idx['<EOS>']]
        
        return {
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "gloss_indices": torch.tensor(gloss_indices, dtype=torch.long),
            "text_indices": torch.tensor(text_indices, dtype=torch.long),
            "gloss_raw": glosses,
            "text_raw": translation_text
        }

def collate_fn(batch):
    """
    Collate function to pad variable-length sequences.
    """
    # 1. Pad landmarks
    landmarks = [item['landmarks'] for item in batch]
    landmark_lens = [len(x) for x in landmarks]
    max_landmark_len = max(landmark_lens)
    input_dim = landmarks[0].shape[1]
    
    padded_landmarks = torch.zeros(len(batch), max_landmark_len, input_dim)
    # Boolean mask: True means masked (padded)
    landmark_mask = torch.ones(len(batch), max_landmark_len, dtype=torch.bool)
    
    for i, seq in enumerate(landmarks):
        padded_landmarks[i, :len(seq), :] = seq
        landmark_mask[i, :len(seq)] = False
        
    # 2. Pad glosses (useful for metrics/translation input)
    gloss_indices = [item['gloss_indices'] for item in batch]
    gloss_lens = [len(x) for x in gloss_indices]
    max_gloss_len = max(gloss_lens)
    
    padded_glosses = torch.zeros(len(batch), max_gloss_len, dtype=torch.long)
    gloss_mask = torch.ones(len(batch), max_gloss_len, dtype=torch.bool)
    
    for i, seq in enumerate(gloss_indices):
        padded_glosses[i, :len(seq)] = seq
        gloss_mask[i, :len(seq)] = False

    # 3. Pad translation text
    text_indices = [item['text_indices'] for item in batch]
    text_lens = [len(x) for x in text_indices]
    max_text_len = max(text_lens)
    
    padded_text = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    text_mask = torch.ones(len(batch), max_text_len, dtype=torch.bool)
    
    for i, seq in enumerate(text_indices):
        padded_text[i, :len(seq)] = seq
        text_mask[i, :len(seq)] = False
        
    return {
        "landmarks": padded_landmarks,
        "landmark_lens": torch.tensor(landmark_lens, dtype=torch.long),
        "landmark_mask": landmark_mask,
        
        "gloss_indices": padded_glosses,
        "gloss_lens": torch.tensor(gloss_lens, dtype=torch.long),
        "gloss_mask": gloss_mask,
        
        "text_indices": padded_text,
        "text_lens": torch.tensor(text_lens, dtype=torch.long),
        "text_mask": text_mask,
        
        "gloss_raw": [item['gloss_raw'] for item in batch],
        "text_raw": [item['text_raw'] for item in batch]
    }
