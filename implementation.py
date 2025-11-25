"""
BacLABNet: Deep Learning Neural Network for Bacteriocin Classification
Reproduces the methodology from González et al. (2025)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import time
import warnings
warnings.filterwarnings('ignore')

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
# 2. K-MER FEATURE EXTRACTION
# ============================================================================

def generate_kmers(sequence: str, k: int) -> List[str]:
    """Generate all k-mers from a sequence"""
    sequence = sequence.upper()
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def get_top_kmers(sequences: List[str], k: int, top_n: int = 100) -> List[str]:
    """Get top N most frequent k-mers from sequences"""
    kmer_counts = {}
    
    for seq in sequences:
        kmers = generate_kmers(seq, k)
        for kmer in kmers:
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    
    # Sort by frequency and get top N
    sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)
    return [kmer for kmer, count in sorted_kmers[:top_n]]

def extract_kmer_features(sequence: str, kmer_list: List[str], k: int) -> np.ndarray:
    """Extract binary k-mer features (presence/absence)"""
    seq_kmers = set(generate_kmers(sequence, k))
    features = np.array([1 if kmer in seq_kmers else 0 for kmer in kmer_list])
    return features


# ============================================================================
# 3. EMBEDDING VECTOR EXTRACTION (GRU-BASED RNN)
# ============================================================================

class EmbeddingRNN(nn.Module):
    """GRU-based RNN for generating embedding vectors"""
    
    def __init__(self, vocab_size: int = 21, embedding_dim: int = 128, 
                 hidden_dim: int = 128):
        super(EmbeddingRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        _, hidden = self.gru(embedded)  # hidden: (1, batch, hidden_dim)
        output = self.linear(hidden.squeeze(0))  # (batch, embedding_dim)
        return output

def extract_embedding_features(sequences: List[str], 
                               embedding_model: EmbeddingRNN,
                               max_len: int = 600,
                               batch_size: int = 64,
                               device: str = 'cpu') -> np.ndarray:
    """
    Extract embedding vectors using pre-trained RNN with batch processing.
    
    Args:
        sequences: List of amino acid sequences
        embedding_model: Pre-trained RNN model
        max_len: Maximum sequence length (reduced from 2000 to 600 for speed)
        batch_size: Number of sequences to process at once (default 64)
        device: Device to run model on ('cpu' or 'cuda')
    
    Returns:
        Array of embedding vectors (n_sequences x embedding_dim)
    """
    embedding_model.eval()
    embedding_model = embedding_model.to(device)
    embeddings = []
    
    # Pre-encode and pad all sequences
    encoded_sequences = []
    for seq in sequences:
        encoded = encode_sequence(seq)
        # Pad or truncate to max_len
        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        encoded_sequences.append(encoded)
    
    # Process in batches for 6-12x speedup
    with torch.no_grad():
        for i in range(0, len(encoded_sequences), batch_size):
            batch = encoded_sequences[i:i + batch_size]
            # Convert batch to tensor
            batch_tensor = torch.LongTensor(batch).to(device)
            # Get embeddings for entire batch at once
            batch_embeddings = embedding_model(batch_tensor).cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


# ============================================================================
# 4. DEEP NEURAL NETWORK ARCHITECTURE
# ============================================================================

class BacteriocinClassifierDNN(nn.Module):
    """Deep Neural Network for Bacteriocin Classification"""
    
    def __init__(self, input_dim: int):
        super(BacteriocinClassifierDNN, self).__init__()
        
        # Block 1: 128 neurons
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Block 2: 64 neurons
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Block 3: 32 neurons
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Block 4: Output (2 neurons)
        self.block4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# ============================================================================
# 5. DATASET CLASS
# ============================================================================

class BacteriocinDataset(Dataset):
    """PyTorch Dataset for bacteriocin sequences"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# 6. TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


# ============================================================================
# 7. K-FOLD CROSS VALIDATION
# ============================================================================

def kfold_cross_validation(features: np.ndarray, 
                          labels: np.ndarray,
                          k: int = 30,
                          epochs: int = 75,
                          batch_size: int = 40,
                          learning_rate: float = 2.5e-5,
                          device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Perform k-fold cross validation"""
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{k}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Create datasets and dataloaders
        train_dataset = BacteriocinDataset(X_train, y_train)
        val_dataset = BacteriocinDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = BacteriocinClassifierDNN(input_dim=features.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        # Final evaluation
        final_loss, final_acc, final_prec, final_rec, final_f1, preds, true_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        fold_results.append({
            'fold': fold + 1,
            'loss': final_loss * 100,  # Convert to percentage
            'accuracy': final_acc * 100,
            'precision': final_prec,
            'recall': final_rec,
            'f1_score': final_f1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'predictions': preds,
            'true_labels': true_labels,
            'model_state': model.state_dict()
        })
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Loss: {final_loss * 100:.2f}%")
        print(f"Accuracy: {final_acc * 100:.2f}%")
        print(f"Precision: {final_prec:.4f}")
        print(f"Recall: {final_rec:.4f}")
        print(f"F1 Score: {final_f1:.4f}")
    
    return fold_results


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def plot_training_curves(fold_results: List[Dict], best_fold_idx: int):
    """Plot training curves for the best fold"""
    best_fold = fold_results[best_fold_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(best_fold['train_losses']) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, [acc * 100 for acc in best_fold['val_accuracies']], 
             'b-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Accuracy - Fold {best_fold["fold"]}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, best_fold['train_losses'], 'r-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, best_fold['val_losses'], 'b-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Loss - Fold {best_fold["fold"]}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(fold_results: List[Dict], best_fold_idx: int):
    """Plot confusion matrix for the best fold"""
    best_fold = fold_results[best_fold_idx]
    
    cm = confusion_matrix(best_fold['true_labels'], best_fold['predictions'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Non-BacLAB', 'BacLAB'],
                yticklabels=['Non-BacLAB', 'BacLAB'])
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title(f'Confusion Matrix (Counts) - Fold {best_fold["fold"]}', 
                  fontsize=14, fontweight='bold')
    
    # Normalized
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=['Non-BacLAB', 'BacLAB'],
                yticklabels=['Non-BacLAB', 'BacLAB'])
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title(f'Confusion Matrix (Normalized) - Fold {best_fold["fold"]}', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline
    
    PERFORMANCE OPTIMIZATIONS:
    - Batch embedding extraction (64 sequences at once) → 6-12x speedup
    - Reduced max_len from 2000 to 600 (bacteriocins are typically 50-150 aa)
    - Pre-trained RNN weights loaded from rnn_gru.pt
    - GPU support: automatically uses CUDA if available
    
    FOR FASTEST RESULTS:
    - Run on Google Colab with free GPU (T4): ~2-5 minutes for embeddings
    - Alternatively: Run locally on Mac CPU: ~4-8 hours total
    """
    
    print("="*70)
    print("BacLABNet: Bacteriocin Classification using Deep Learning")
    print("="*70)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    # CSV has no headers: ID, Species, Sequence, Label, Empty
    df_baclabnonbaclabdata = pd.read_csv('data_BacLAB_and_nonBacLAB.csv', 
                                          header=None, 
                                          names=['ID', 'Species', 'Sequence', 'Label', 'Empty'])
    
    sequences = df_baclabnonbaclabdata['Sequence'].tolist()
    labels = df_baclabnonbaclabdata['Label'].values  # 1 for BacLAB, 0 for Non-BacLAB
    
    print(f"Total sequences: {len(sequences)}")
    print(f"BacLAB: {sum(labels)}, Non-BacLAB: {len(labels) - sum(labels)}")
    
    # 2. Extract k-mer features
    print("\n[2/7] Extracting k-mer features...")
    
    # For reproduction, load pre-computed k-mers from file
    try:
        kmers_df = pd.read_csv('List_kmers.csv')
        kmers_5 = kmers_df['5-mers'].dropna().tolist()
        kmers_7 = kmers_df['7-mers'].dropna().tolist()
        print("Loaded pre-computed k-mers from file")
    except:
        print("Computing k-mers from BacLAB sequences...")
        baclab_sequences = [seq for seq, lbl in zip(sequences, labels) if lbl == 1]
        kmers_5 = get_top_kmers(baclab_sequences, k=5, top_n=100)
        kmers_7 = get_top_kmers(baclab_sequences, k=7, top_n=100)
    
    # Extract k-mer features for all sequences
    features_5 = np.array([extract_kmer_features(seq, kmers_5, 5) for seq in sequences])
    features_7 = np.array([extract_kmer_features(seq, kmers_7, 7) for seq in sequences])
    
    print(f"5-mer features shape: {features_5.shape}")
    print(f"7-mer features shape: {features_7.shape}")
    
    # 3. Extract embedding features
    print("\n[3/7] Extracting embedding features...")
    
    # Try to load pre-computed embeddings (from Colab GPU run)
    try:
        embedding_features = np.load('embeddings.npy')
        print("✓ Loaded pre-computed embeddings from 'embeddings.npy'")
        print(f"  Shape: {embedding_features.shape}")
        print("  (Skipping embedding extraction - using cached results)")
    except FileNotFoundError:
        print("No pre-computed embeddings found. Extracting embeddings...")
        print("  TIP: For faster processing, run 'colab_embedding_extraction.py' on Google Colab GPU")
        
        # Detect device (GPU if available, otherwise CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device.upper()}")
        
        # Initialize embedding model
        embedding_model = EmbeddingRNN(vocab_size=21, embedding_dim=128)
        
        # Load pre-trained RNN weights from published model
        try:
            state_dict = torch.load('rnn_gru.pt', map_location=device)
            embedding_model.load_state_dict(state_dict)
            print("✓ Loaded pre-trained RNN model from 'rnn_gru.pt'")
        except Exception as e:
            print(f"⚠ Warning: Could not load pre-trained model: {e}")
            print("  Using randomly initialized embedding model instead")
        
        # Extract embeddings with batching (6-12x faster than 1-by-1)
        start_time = time.time()
        embedding_features = extract_embedding_features(
            sequences, 
            embedding_model, 
            max_len=600,  # Reduced from 2000 (most bacteriocins are 50-150 aa)
            batch_size=64,  # Process 64 sequences at once
            device=device
        )
        elapsed_time = time.time() - start_time
        print(f"Embedding features shape: {embedding_features.shape}")
        print(f"Extraction time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Speed: {len(sequences)/elapsed_time:.1f} sequences/second")
        
        # Save for future use
        np.save('embeddings.npy', embedding_features)
        print("✓ Saved embeddings to 'embeddings.npy' for future runs")
    
    # 4. Concatenate features (5-mers + 7-mers + EV)
    print("\n[4/7] Concatenating features...")
    features = np.concatenate([features_5, features_7, embedding_features], axis=1)
    print(f"Final features shape: {features.shape}")
    print(f"Input dimension: {features.shape[1]} (100 + 100 + 128 = 328)")
    
    # 5. Perform k-fold cross-validation
    print("\n[5/7] Starting k-fold cross-validation (k=30)...")
    fold_results = kfold_cross_validation(
        features=features,
        labels=labels,
        k=30,
        epochs=75,
        batch_size=40,
        learning_rate=2.5e-5
    )
    
    # 6. Analyze results
    print("\n[6/7] Analyzing results...")
    
    results_df = pd.DataFrame([{
        'Fold': r['fold'],
        'Loss (%)': r['loss'],
        'Accuracy (%)': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1 Score': r['f1_score']
    } for r in fold_results])
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS (k=30)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("AVERAGE METRICS")
    print("="*70)
    print(f"Loss: {results_df['Loss (%)'].mean():.2f}%")
    print(f"Accuracy: {results_df['Accuracy (%)'].mean():.2f}%")
    print(f"Precision: {results_df['Precision'].mean():.4f}")
    print(f"Recall: {results_df['Recall'].mean():.4f}")
    print(f"F1 Score: {results_df['F1 Score'].mean():.4f}")
    
    # Find best fold (Fold 22 in paper)
    best_fold_idx = results_df['Accuracy (%)'].idxmax()
    best_fold = fold_results[best_fold_idx]
    
    print("\n" + "="*70)
    print(f"BEST FOLD: Fold {best_fold['fold']}")
    print("="*70)
    print(f"Loss: {best_fold['loss']:.2f}%")
    print(f"Accuracy: {best_fold['accuracy']:.2f}%")
    print(f"Precision: {best_fold['precision']:.4f}")
    print(f"Recall: {best_fold['recall']:.4f}")
    print(f"F1 Score: {best_fold['f1_score']:.4f}")
    
    # 7. Visualize results
    print("\n[7/7] Generating visualizations...")
    plot_training_curves(fold_results, best_fold_idx)
    plot_confusion_matrix(fold_results, best_fold_idx)
    
    # Save best model
    torch.save(best_fold['model_state'], 'best_model_fold22.pt')
    print("\nBest model saved as 'best_model_fold22.pt'")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()