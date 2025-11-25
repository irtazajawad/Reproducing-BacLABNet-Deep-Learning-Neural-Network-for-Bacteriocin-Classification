"""
BacLABNet: Fast Embedding Extraction for Google Colab GPU
===========================================================

This script extracts ONLY the embedding features using GPU acceleration.
Run this on Google Colab with GPU enabled for ~2-5 minutes processing time.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List
import time

# ============================================================================
# 1. AMINO ACID ENCODING
# ============================================================================

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

AA_TO_IDX = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['X'] = 0  # Unknown amino acid

def encode_sequence(sequence: str) -> List[int]:
    """Encode amino acid sequence to integer indices"""
    return [AA_TO_IDX.get(aa, 0) for aa in sequence.upper()]


# ============================================================================
# 2. EMBEDDING RNN MODEL
# ============================================================================

class EmbeddingRNN(nn.Module):
    """GRU-based RNN for generating embedding vectors (matches rnn_gru.pt architecture)"""
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 10, 
                 hidden_dim: int = 128):
        super(EmbeddingRNN, self).__init__()
        
        # Architecture from rnn_gru.pt checkpoint:
        # - embedding: [21, 10] - small embedding dimension
        # - GRU: input=10, hidden=128
        # - decoder: [21, 128] - predicts next amino acid (language model)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)  # Decoder outputs vocab_size (21)
        
    def forward(self, x, return_embedding=True):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, 10)
        _, hidden = self.gru(embedded)  # hidden: (1, batch, 128)
        
        if return_embedding:
            # Return the 128-dim GRU hidden state as embedding
            return hidden.squeeze(0)  # (batch, 128)
        else:
            # Return decoder output for language modeling
            output = self.decoder(hidden.squeeze(0))  # (batch, 21)
            return output


# ============================================================================
# 3. FAST BATCH EMBEDDING EXTRACTION
# ============================================================================

def extract_embedding_features_gpu(sequences: List[str], 
                                   embedding_model: EmbeddingRNN,
                                   max_len: int = 600,
                                   batch_size: int = 128,  # Larger batch for GPU
                                   device: str = 'cuda') -> np.ndarray:
    """
    Extract embedding vectors using GPU with large batch processing.
    
    Args:
        sequences: List of amino acid sequences
        embedding_model: Pre-trained RNN model
        max_len: Maximum sequence length (600 covers 99%+ of bacteriocins)
        batch_size: 128 for GPU (vs 64 for CPU)
        device: 'cuda' for GPU
    
    Returns:
        Array of embedding vectors (n_sequences x 128)
    """
    embedding_model.eval()
    embedding_model = embedding_model.to(device)
    embeddings = []
    
    print(f"Encoding {len(sequences)} sequences...")
    # Pre-encode and pad all sequences
    encoded_sequences = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 10000 == 0:
            print(f"  Encoded {i + 1}/{len(sequences)} sequences...")
        encoded = encode_sequence(seq)
        # Pad or truncate to max_len
        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        encoded_sequences.append(encoded)
    
    print(f"\nExtracting embeddings on {device.upper()}...")
    # Process in batches
    with torch.no_grad():
        for i in range(0, len(encoded_sequences), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"  Processed {i}/{len(encoded_sequences)} sequences...")
            batch = encoded_sequences[i:i + batch_size]
            batch_tensor = torch.LongTensor(batch).to(device)
            batch_embeddings = embedding_model(batch_tensor).cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


# ============================================================================
# 4. MAIN SCRIPT
# ============================================================================

def main():
    print("="*70)
    print("BacLABNet: GPU-Accelerated Embedding Extraction")
    print("="*70)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device.upper()}")
    if device == 'cpu':
        print("  ⚠ WARNING: GPU not detected! This will be slow.")
        print("  → Enable GPU: Runtime → Change runtime type → T4 GPU")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
    
    # Load data
    print("\n[1/3] Loading sequences...")
    df = pd.read_csv('data_BacLAB_and_nonBacLAB.csv', 
                     header=None, 
                     names=['ID', 'Species', 'Sequence', 'Label', 'Empty'])
    
    sequences = df['Sequence'].tolist()
    labels = df['Label'].values
    
    print(f"  Total sequences: {len(sequences):,}")
    print(f"  BacLAB: {sum(labels):,}, Non-BacLAB: {len(labels) - sum(labels):,}")
    
    # Calculate sequence length statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
          f"mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.1f}")
    
    # Load pre-trained model
    print("\n[2/3] Loading pre-trained RNN model...")
    embedding_model = EmbeddingRNN(vocab_size=21, embedding_dim=10, hidden_dim=128)
    
    try:
        state_dict = torch.load('rnn_gru.pt', map_location=device)
        embedding_model.load_state_dict(state_dict)
        print("  ✓ Loaded rnn_gru.pt")
        print("     Model architecture: embedding(21→10) → GRU(10→128) → decoder(128→21)")
        print("     Using GRU hidden state (128-dim) as protein embeddings")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return
    
    # Extract embeddings
    print("\n[3/3] Extracting embeddings...")
    start_time = time.time()
    
    embedding_features = extract_embedding_features_gpu(
        sequences, 
        embedding_model, 
        max_len=600,
        batch_size=128 if device == 'cuda' else 64,
        device=device
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Complete!")
    print(f"  Shape: {embedding_features.shape}")
    print(f"  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  Speed: {len(sequences)/elapsed_time:.1f} sequences/second")
    
    # Save embeddings
    print("\n[4/3] Saving embeddings...")
    np.save('embeddings.npy', embedding_features)
    print("  ✓ Saved to: embeddings.npy")
    print(f"  File size: {embedding_features.nbytes / (1024**2):.2f} MB")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Download embeddings.npy from Colab")
    print("2. Copy to your local project directory")
    print("3. Run the main training script (it will load pre-computed embeddings)")
    print("="*70)


if __name__ == "__main__":
    main()

