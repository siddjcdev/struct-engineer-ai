import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================================================================
# 1. LOAD AND PREPROCESS PEER EARTHQUAKE DATASET
# ============================================================================

def load_peer_dataset(data_file):
    """Load PEER earthquake dataset from JSON"""
    with open(data_file, 'r') as f:
        earthquakes = json.load(f)
    return earthquakes

def inspect_dataset(earthquakes):
    """Inspect dataset structure"""
    print("=== Dataset Inspection ===")
    print(f"Type: {type(earthquakes)}")
    
    if isinstance(earthquakes, dict):
        print(f"Top-level keys: {earthquakes.keys()}")
        
        # Check structure
        if 'input_features' in earthquakes:
            print(f"\ninput_features type: {type(earthquakes['input_features'])}")
            print(f"input_features shape: {len(earthquakes['input_features']) if isinstance(earthquakes['input_features'], list) else 'N/A'}")
            if isinstance(earthquakes['input_features'], list) and len(earthquakes['input_features']) > 0:
                print(f"First input sample: {earthquakes['input_features'][0]}")
        
        if 'output_feature' in earthquakes:
            print(f"\noutput_feature type: {type(earthquakes['output_feature'])}")
            print(f"output_feature shape: {len(earthquakes['output_feature']) if isinstance(earthquakes['output_feature'], list) else 'N/A'}")
            if isinstance(earthquakes['output_feature'], list) and len(earthquakes['output_feature']) > 0:
                print(f"First output sample: {earthquakes['output_feature'][0]}")
        
        if 'building' in earthquakes:
            print(f"\nbuilding: {earthquakes['building']}")
        
        if 'n_samples' in earthquakes:
            print(f"n_samples: {earthquakes['n_samples']}")
        
        print(f"\nFull structure (first 800 chars):\n{json.dumps(earthquakes, indent=2)[:800]}")

def preprocess_data(earthquakes):
    """Convert earthquake data to training sequences - handle dict with 'samples' and keys
    like 'displacement','velocity','force'."""
    X_list = []
    y_list = []

    # Normalize samples
    if isinstance(earthquakes, dict):
        if 'samples' in earthquakes:
            samples = earthquakes['samples']
            if isinstance(samples, dict):
                samples = list(samples.values())
        elif 'input_features' in earthquakes and 'output_feature' in earthquakes:
            return np.array(earthquakes['input_features']), np.array(earthquakes['output_feature'])
        else:
            raise ValueError(f"Unexpected data format: {earthquakes.keys()}")
    else:
        samples = earthquakes

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("No samples found in dataset")

    # If sample keys are displacement/velocity/force, handle explicitly
    first = samples[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        if {'displacement', 'velocity', 'force'}.issubset(set(keys)):
            for s in samples:
                disp = np.atleast_1d(np.asarray(s.get('displacement', []), dtype=float))
                vel = np.atleast_1d(np.asarray(s.get('velocity', []), dtype=float))
                force = s.get('force')

                # force may be scalar or array
                force_arr = np.atleast_1d(np.asarray(force, dtype=float))

                # if disp/vel are sequences, build per-time-step samples
                n = min(len(disp), len(vel), len(force_arr)) if (disp.size > 1 or vel.size > 1 or force_arr.size > 1) else 1
                if n == 0:
                    continue

                if n == 1:
                    X_list.append([float(disp.ravel()[0]), float(vel.ravel()[0])])
                    y_list.append([float(force_arr.ravel()[0])])
                else:
                    for i in range(n):
                        X_list.append([float(disp.ravel()[i]), float(vel.ravel()[i])])
                        y_list.append([float(force_arr.ravel()[i])])

            if len(X_list) == 0:
                raise ValueError("No valid displacement/velocity/force samples extracted")

            X = np.asarray(X_list, dtype=float)
            y = np.asarray(y_list, dtype=float)
            print(f"✅ Extracted {X.shape[0]} samples from displacement/velocity/force keys")
            return X, y

    # Fallback: previous generic detection/extraction
    input_key = None
    output_key = None
    input_candidates = ['input_features', 'input', 'inputs', 'features', 'state']
    output_candidates = ['output_feature', 'output', 'target', 'labels', 'force']

    if isinstance(first, dict):
        for k in input_candidates:
            if k in first:
                input_key = k
                break
        for k in output_candidates:
            if k in first:
                output_key = k
                break

    if input_key is None or output_key is None:
        # try to infer by types
        for k, v in (first.items() if isinstance(first, dict) else []):
            if input_key is None and (isinstance(v, list) or (hasattr(v, '__len__') and np.asarray(v).ndim >= 1)):
                input_key = k
            elif output_key is None and (isinstance(v, (int, float)) or (isinstance(v, list) and np.asarray(v).size == 1)):
                output_key = k
            if input_key and output_key:
                break

    if input_key is None or output_key is None:
        raise ValueError(f"Could not detect input/output keys. Sample keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}")

    X = []
    y = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        inp = s.get(input_key)
        out = s.get(output_key)
        if inp is None or out is None:
            continue
        X.append(np.asarray(inp, dtype=float).ravel())
        y.append(np.asarray(out, dtype=float).ravel())

    if len(X) == 0:
        raise ValueError("No valid input/output pairs extracted from samples")

    X = np.vstack(X) if np.asarray(X).ndim > 1 else np.asarray(X)
    y = np.vstack(y) if np.asarray(y).ndim > 1 else np.asarray(y)

    print(f"✅ Extracted {X.shape[0]} samples. input_key='{input_key}' output_key='{output_key}'")
    return X, y

# ============================================================================
# 2. DEFINE RL-BASED POLICY NETWORK
# ============================================================================

class TMDPolicyNetwork(nn.Module):
    """Policy network for TMD control (Actor)"""
    
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64):
        super(TMDPolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, action_dim)
        )
        
        # Output scaling (tanh for bounded control force)
        self.output_scale = 1.0
    
    def forward(self, state):
        """Forward pass: state (disp, vel) -> control force"""
        action = self.network(state)
        return torch.tanh(action) * self.output_scale

class TMDValueNetwork(nn.Module):
    """Value network for RL baseline (Critic)"""
    
    def __init__(self, state_dim=2, hidden_dim=64):
        super(TMDValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, state):
        """Forward pass: state -> value estimate"""
        return self.network(state)

# ============================================================================
# 3. ACTOR-CRITIC RL TRAINER
# ============================================================================

class RLTMDTrainer:
    """Trainer for RL-based TMD control"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Networks
        self.policy = TMDPolicyNetwork().to(device)
        self.value = TMDValueNetwork().to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)
        
        # Loss
        self.mse_loss = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def compute_advantage(self, states, actions, rewards, next_states, gamma=0.99):
        """Compute advantage for policy gradient"""
        with torch.no_grad():
            current_values = self.value(states).squeeze()
            next_values = self.value(next_states).squeeze()
            
            # TD error: A = r + gamma * V(s') - V(s)
            td_target = rewards + gamma * next_values
            advantages = td_target - current_values
        
        return advantages, td_target
    
    def train_step(self, states, actions, rewards, next_states):
        """Single training step"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Compute advantages
        advantages, td_targets = self.compute_advantage(states, actions, rewards, next_states)
        
        # Policy loss (actor): minimize negative log probability * advantage
        predicted_actions = self.policy(states)
        policy_loss = -(torch.log(torch.clamp(predicted_actions, 1e-5, 1-1e-5)) * advantages.detach()).mean()
        
        # Value loss (critic): MSE between predicted and target value
        predicted_values = self.value(states).squeeze()
        value_loss = self.mse_loss(predicted_values, td_targets.detach())
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return (policy_loss + value_loss).item()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Full training loop with edge case handling"""
        
        # Validate data
        if len(X_train) == 0:
            raise ValueError("X_train is empty!")
        if len(X_val) == 0:
            print("⚠️  Warning: X_val is empty. Using last 10% of training data for validation.")
            split_idx = int(0.9 * len(X_train))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = max(1, len(X_train) // batch_size)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                states = X_train[start_idx:end_idx]
                actions = y_train[start_idx:end_idx]
                
                # Simulated rewards: negative control effort (minimize force)
                rewards = -np.abs(actions).squeeze()
                
                # Next states (shifted by 1, pad if necessary)
                if end_idx < len(X_train):
                    next_states = X_train[end_idx:min(end_idx + len(states), len(X_train))]
                else:
                    next_states = X_train[-len(states):]
                
                # Pad if needed
                if len(next_states) < len(states):
                    pad_size = len(states) - len(next_states)
                    next_states = np.vstack([next_states, np.tile(X_train[-1:], (pad_size, 1))])
                
                loss = self.train_step(states, actions, rewards, next_states[:len(states)])
                epoch_loss += loss
            
            # Validation (check if X_val has samples)
            if len(X_val) > 0:
                val_loss = self.validate(X_val, y_val)
            else:
                val_loss = 0
            
            self.train_losses.append(epoch_loss / num_batches)
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {self.train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}")
    
    def validate(self, X_val, y_val):
        """Validation step"""
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        with torch.no_grad():
            predictions = self.policy(X_val)
            loss = self.mse_loss(predictions, y_val)
        
        return loss.item()
    
    def predict(self, state):
        """Predict control force for a given state"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.policy(state)
        return action.cpu().numpy()
    
    def save_model(self, filepath):
        """Save trained models"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RL TMD Control Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ============================================================================
# 4. MAIN TRAINING PIPELINE
# ============================================================================

if __name__ == "__main__":
    # Use the correct dataset filename
    data_file = Path("../tmd_training_data_peer.json")
    
    # Try multiple path variations
    possible_paths = [
        Path("tmd_training_data_peer.json"),
        Path("../../tmd_training_data_peer.json"),
        Path("./tmd_training_data_peer.json"),
        Path("c:/Dev/dAmpIng26/git/struct-engineer-ai/neuralnet/src/tmd_training_data_peer.json"),
        Path("c:/Dev/dAmpIng26/git/struct-engineer-ai/tmd_training_data_peer.json"),
    ]
    
    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break
    
    if not data_file:
        print("❌ Dataset 'tmd_training_data_peer.json' not found.")
        print(f"\nCurrent directory: {Path.cwd()}")
        print("\nSearching for the file...")
        found_files = list(Path("c:/Dev/dAmpIng26/git/struct-engineer-ai").rglob("tmd_training_data_peer.json"))
        if found_files:
            data_file = found_files[0]
            print(f"✅ Found at: {data_file}")
        else:
            print("❌ File not found anywhere in project.")
            exit(1)
    
    print(f"✅ Loading dataset from: {data_file}\n")
    earthquakes = load_peer_dataset(data_file)
    
    # INSPECT FIRST
    inspect_dataset(earthquakes)
    
    # Preprocess
    print("\n=== Preprocessing Data ===")
    try:
        X, y = preprocess_data(earthquakes)
        print(f"✅ Dataset shape: X={X.shape}, y={y.shape}")
        print(f"Sample X[0]: {X[0]}")
        print(f"Sample y[0]: {y[0]}\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # Train-val split
    if len(X) < 10:
        print(f"⚠️  Only {len(X)} samples. Reducing validation split.")
        split_idx = max(1, int(0.7 * len(X)))
    else:
        split_idx = int(0.8 * len(X))
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {X_train.shape} | Val set: {X_val.shape}\n")
    
    # Initialize trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    trainer = RLTMDTrainer(device=device)
    
    # Train
    print("Training RL TMD model...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Save model
    trainer.save_model("tmd_rl_model.pt")
    
    # Plot results
    trainer.plot_losses()
    
    # Test prediction
    print("\n=== Test Prediction ===")
    test_state = np.array([[0.00012358745738059112, 0.006522938179246303]])
    control_force = trainer.predict(test_state)
    print(f"State: displacement=0.0001236m, velocity=0.006523m/s")
    print(f"Predicted control force: {control_force[0][0]:.4f} kN")