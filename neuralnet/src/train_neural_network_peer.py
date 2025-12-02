"""
Train Neural Network TMD Controller on PEER Earthquake Data
Complete training pipeline with visualization and model export
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict


class TMDNeuralNetwork(nn.Module):
    """
    Neural Network for TMD Control
    Input: [displacement, velocity]
    Output: [control_force]
    """
    
    def __init__(self):
        super(TMDNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer: 2 features (displacement, velocity)
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 1
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # Output layer: 1 value (control force)
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class TMDDataset(Dataset):
    """PyTorch Dataset for TMD training data"""
    
    def __init__(self, data_list):
        # Convert to numpy arrays
        self.X = np.array([[d, v] for d, v, f in data_list], dtype=np.float32)
        self.y = np.array([[f] for d, v, f in data_list], dtype=np.float32)
        
        # Calculate normalization parameters
        self.input_mean = self.X.mean(axis=0)
        self.input_std = self.X.std(axis=0) + 1e-8
        self.output_mean = self.y.mean(axis=0)
        self.output_std = self.y.std(axis=0) + 1e-8
        
        # Store normalization for later use
        self.normalization = {
            'input_mean': self.input_mean.tolist(),
            'input_std': self.input_std.tolist(),
            'output_mean': self.output_mean.tolist(),
            'output_std': self.output_std.tolist()
        }
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return normalized data
        x_norm = (self.X[idx] - self.input_mean) / self.input_std
        y_norm = (self.y[idx] - self.output_mean) / self.output_std
        return torch.tensor(x_norm), torch.tensor(y_norm)


class NeuralTMDController:
    """
    Wrapper class for trained neural network controller
    Handles normalization and provides simple interface
    """
    
    def __init__(self, model_path: str = None):
        self.model = TMDNeuralNetwork()
        self.normalization = None
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def compute(self, displacement: float, velocity: float) -> float:
        """
        Compute control force
        
        Args:
            displacement: Building displacement (m)
            velocity: Building velocity (m/s)
            
        Returns:
            Control force (kN)
        """
        if self.normalization is None:
            raise ValueError("Model not loaded or trained")
        
        # Normalize inputs
        x = np.array([[displacement, velocity]], dtype=np.float32)
        x_norm = (x - self.normalization['input_mean']) / self.normalization['input_std']
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            # Explicitly convert to float32 tensor to match model dtype
            x_tensor = torch.tensor(x_norm, dtype=torch.float32)
            y_norm = self.model(x_tensor).numpy()
        
        # Denormalize output
        y = y_norm * self.normalization['output_std'] + self.normalization['output_mean']
        
        return float(y[0, 0])
    
    def save(self, filepath: str):
        """Save model and normalization parameters"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'normalization': self.normalization
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and normalization parameters"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.normalization = checkpoint['normalization']
        self.model.eval()
        print(f"✅ Model loaded from {filepath}")


def train_model(
    data_path: str = 'tmd_training_data_peer.json',
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    device: str = 'auto'
) -> Tuple[NeuralTMDController, Dict]:
    """
    Complete training pipeline
    
    Args:
        data_path: Path to training data JSON
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data for validation
        device: 'auto', 'cpu', or 'cuda'
        
    Returns:
        trained_controller: Trained NeuralTMDController
        history: Training history dictionary
    """
    
    print("="*70)
    print("TRAINING NEURAL NETWORK TMD CONTROLLER")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("Loading training data...")
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    
    # Extract samples as list of tuples
    training_data = [
        (s['displacement'], s['velocity'], s['force'])
        for s in data_json['samples']
    ]
    
    print(f"  ✅ Loaded {len(training_data)} samples")
    print(f"  Displacement range: [{min(d for d,v,f in training_data):.3f}, {max(d for d,v,f in training_data):.3f}] m")
    print(f"  Velocity range: [{min(v for d,v,f in training_data):.3f}, {max(v for d,v,f in training_data):.3f}] m/s")
    print(f"  Force range: [{min(f for d,v,f in training_data):.1f}, {max(f for d,v,f in training_data):.1f}] kN")
    print()
    
    # ========================================================================
    # 2. PREPARE DATASETS
    # ========================================================================
    print("Preparing datasets...")
    
    # Create full dataset
    full_dataset = TMDDataset(training_data)
    
    # Split into train/validation
    n_total = len(full_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Copy normalization to validation dataset
    val_dataset.dataset.normalization = full_dataset.normalization
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # ========================================================================
    # 3. SETUP MODEL AND TRAINING
    # ========================================================================
    print("Setting up model...")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    print(f"  Device: {device}")
    
    # Create model
    model = TMDNeuralNetwork().to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # ========================================================================
    # 4. TRAINING LOOP
    # ========================================================================
    print("Starting training...")
    print("-" * 70)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= n_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= n_val
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.3f} | "
                  f"Val Loss: {val_loss:.3f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("-" * 70)
    print(f"✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.3f}")
    print()
    
    # ========================================================================
    # 5. RESTORE BEST MODEL AND CREATE CONTROLLER
    # ========================================================================
    print("Creating controller with best model...")
    model.load_state_dict(best_model_state)
    
    controller = NeuralTMDController()
    controller.model = model.cpu()
    controller.normalization = full_dataset.normalization
    
    print("  ✅ Controller ready")
    print()
    
    # ========================================================================
    # 6. SAVE MODEL
    # ========================================================================
    save_path = 'tmd_trained_model_peer.pth'
    controller.save(save_path)
    print()
    
    # ========================================================================
    # 7. VISUALIZE TRAINING
    # ========================================================================
    print("Generating training visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Learning rate
    axes[0, 1].plot(history['learning_rate'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction vs True (validation set sample)
    model.eval()
    with torch.no_grad():
        # Get a batch from validation
        X_val, y_val = next(iter(val_loader))
        X_val = X_val.to(device)
        y_pred = model(X_val).cpu().numpy()
        y_true = y_val.numpy()
        
        # Denormalize
        y_pred_denorm = y_pred * full_dataset.output_std + full_dataset.output_mean
        y_true_denorm = y_true * full_dataset.output_std + full_dataset.output_mean
    
    axes[1, 0].scatter(y_true_denorm, y_pred_denorm, alpha=0.5, s=10)
    min_val = min(y_true_denorm.min(), y_pred_denorm.min())
    max_val = max(y_true_denorm.max(), y_pred_denorm.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Force (kN)')
    axes[1, 0].set_ylabel('Predicted Force (kN)')
    axes[1, 0].set_title('Prediction Accuracy (Validation Set)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Control surface
    disp_range = np.linspace(-0.3, 0.3, 50)
    vel_range = np.linspace(-1.5, 1.5, 50)
    D, V = np.meshgrid(disp_range, vel_range)
    
    F = np.zeros_like(D)
    for i in range(len(disp_range)):
        for j in range(len(vel_range)):
            F[j, i] = controller.compute(D[j, i], V[j, i])
    
    contour = axes[1, 1].contourf(D, V, F, levels=20, cmap='RdBu_r')
    axes[1, 1].set_xlabel('Displacement (m)')
    axes[1, 1].set_ylabel('Velocity (m/s)')
    axes[1, 1].set_title('Learned Control Surface')
    plt.colorbar(contour, ax=axes[1, 1], label='Control Force (kN)')
    
    plt.tight_layout()
    plt.savefig('training_results_peer.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved training_results_peer.png")
    plt.close()
    
    print()
    print("="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print()
    print("Files created:")
    print(f"  1. {save_path} - Trained model")
    print(f"  2. training_results_peer.png - Training visualizations")
    print()
    print("Test the model:")
    print("  python test_neural_controller.py")
    
    return controller, history


def main():
    """Main execution"""
    # Train the model
    controller, history = train_model(
        data_path='tmd_training_data_peer.json',
        epochs=100,
        batch_size=128,
        learning_rate=0.001,
        validation_split=0.2
    )
    
    # Quick test
    print()
    print("Quick Test:")
    test_cases = [
        (0.1, 0.5),   # Moderate displacement, positive velocity
        (-0.1, -0.5), # Moderate displacement, negative velocity
        (0.0, 0.0),   # At rest
        (0.3, 1.0),   # Large displacement and velocity
    ]
    
    for disp, vel in test_cases:
        force = controller.compute(disp, vel)
        print(f"  Disp: {disp:+.2f}m, Vel: {vel:+.2f}m/s → Force: {force:+.1f} kN")


if __name__ == '__main__':
    main()
