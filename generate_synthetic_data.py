import os
import numpy as np
import yaml
import random

def generate_synthetic_dataset():
    # Load vocabulary from config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    gloss_vocab = config['data']['gloss_vocab'][3:] # Skip special tokens <PAD>, <SOS>, <EOS>
    text_vocab = config['data']['text_vocab'][4:]   # Skip <PAD>, <SOS>, <EOS>, <UNK>
    
    # We define standard sentences/glosses mappings
    sentence_templates = [
        {
            "glosses": ["I", "GO", "UNIVERSITY", "TOMORROW"],
            "translation": "i am going to the university tomorrow ."
        },
        {
            "glosses": ["YOU", "WANT", "LEARN", "SIGN", "LANGUAGE"],
            "translation": "you want to learn sign language ."
        },
        {
            "glosses": ["HELLO", "TODAY", "WE", "LEARN", "SIGN"],
            "translation": "hello today we are learning sign language ."
        },
        {
            "glosses": ["I", "LIKE", "UNIVERSITY", "GOOD"],
            "translation": "i like the university it is good ."
        },
        {
            "glosses": ["YESTERDAY", "YOU", "COME", "UNIVERSITY"],
            "translation": "yesterday you came to the university ."
        },
        {
            "glosses": ["WE", "GO", "TOMORROW", "GOOD"],
            "translation": "we are going tomorrow which is good ."
        },
        {
            "glosses": ["HELLO", "I", "WANT", "COME"],
            "translation": "hello i want to come ."
        },
        {
            "glosses": ["YOU", "LIKE", "LEARN", "LANGUAGE"],
            "translation": "do you like learning languages ."
        }
    ]

    input_dim = config['model']['spatial']['input_dim'] # 225

    # Create directories
    for split in ['train', 'val']:
        split_dir = os.path.join("data", "processed", split)
        os.makedirs(split_dir, exist_ok=True)
        
        num_samples = 120 if split == 'train' else 30
        print(f"Generating {num_samples} synthetic samples for {split}...")

        for i in range(num_samples):
            # Select a template
            template = random.choice(sentence_templates)
            glosses = template["glosses"]
            translation = template["translation"]
            
            # Determine length of video frames based on number of glosses
            num_glosses = len(glosses)
            num_frames = num_glosses * 30 + random.randint(-15, 15)
            num_frames = max(30, num_frames)

            # Generate dummy landmark coordinates [num_frames, 225]
            # Hand landmarks are often 0 if hands are not visible, but let's simulate hands moving
            landmarks = np.zeros((num_frames, input_dim), dtype=np.float32)
            
            # Simulate pose coordinates (first 99 values) - slightly vibrating around a center
            pose_center = np.random.uniform(0.3, 0.7, 99)
            # Simulate left hand (next 63 values) and right hand (last 63 values)
            lh_center = np.random.uniform(0.2, 0.5, 63)
            rh_center = np.random.uniform(0.5, 0.8, 63)
            
            # Create a smooth trajectory using sine/cosine for motion
            for f in range(num_frames):
                t_val = f / num_frames * 2 * np.pi
                pose_offset = 0.02 * np.sin(t_val)
                lh_offset = 0.1 * np.cos(t_val * num_glosses)
                rh_offset = 0.1 * np.sin(t_val * num_glosses)
                
                landmarks[f, 0:99] = pose_center + pose_offset
                landmarks[f, 99:162] = lh_center + lh_offset
                landmarks[f, 162:225] = rh_center + rh_offset
                
            # Save sample
            np.savez(
                os.path.join(split_dir, f"sample_{i:04d}.npz"),
                landmarks=landmarks,
                glosses=glosses,
                translation=translation
            )

    print("Synthetic dataset generation completed successfully.")

if __name__ == "__main__":
    generate_synthetic_dataset()
