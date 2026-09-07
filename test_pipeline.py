import os
import torch
import numpy as np
from inference.predictor import SignLanguagePredictor

def test_pipeline():
    print("Testing end-to-end translation pipeline...")
    
    # Initialize predictor
    predictor = SignLanguagePredictor()
    status = predictor.get_status()
    print(f"Predictor Status: {status}")
    
    # Load a sample from the validation set
    val_dir = os.path.join("data", "processed", "val")
    samples = [f for f in os.listdir(val_dir) if f.endswith(".npz")]
    
    if len(samples) == 0:
        print("Error: No validation samples found.")
        return
        
    sample_path = os.path.join(val_dir, samples[0])
    print(f"Loading validation sample: {sample_path}")
    
    data = np.load(sample_path, allow_pickle=True)
    landmarks = data['landmarks']
    ref_glosses = data['glosses'].tolist()
    ref_translation = str(data['translation'])
    
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Ground Truth Glosses: {ref_glosses}")
    print(f"Ground Truth Translation: '{ref_translation}'")
    
    # Predict directly from landmarks
    pred_glosses, pred_translation, confidence = predictor.predict(landmarks)
    
    print("\n--- Model Outputs (From Landmarks) ---")
    print(f"Predicted Glosses: {pred_glosses}")
    print(f"Predicted Translation: '{pred_translation}'")
    print(f"Confidence Score: {confidence:.4f}")
    
    # Direct Translation Test: bypass recognition model to test the translation model
    print("\n--- Direct Translation Model Test ---")
    print(f"Feeding Ground Truth Glosses to Translation Model: {ref_glosses}")
    
    # Create input indices
    gloss_indices = [predictor.gloss_to_idx['<SOS>']] + [predictor.gloss_to_idx[g] for g in ref_glosses] + [predictor.gloss_to_idx['<EOS>']]
    gloss_tensor = torch.tensor([gloss_indices], dtype=torch.long, device=predictor.device)
    
    # Translate
    translated_ids = predictor.translation_model.translate(
        src=gloss_tensor,
        src_vocab=predictor.gloss_vocab,
        tgt_vocab=predictor.text_vocab,
        device=predictor.device
    )
    
    words = []
    for idx in translated_ids:
        word = predictor.text_vocab[idx]
        if word not in ["<PAD>", "<SOS>", "<EOS>"]:
            words.append(word)
            
    translation_str = " ".join(words)
    print(f"Translated English Output: '{translation_str}'")
    
    print("\nPipeline test completed successfully!")

if __name__ == "__main__":
    test_pipeline()
