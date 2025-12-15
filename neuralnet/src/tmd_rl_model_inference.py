import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class TMDPolicyNetwork(nn.Module):
    """Policy network architecture matching training script"""
    def __init__(self, state_dim=2, action_dim=1, hidden_dim=64, output_scale=1.0):
        super().__init__()
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
        self.output_scale = output_scale

    def forward(self, state):
        action = self.network(state)
        return torch.tanh(action) * self.output_scale

class RLTMDInference:
    """Load trained actor (policy) and run inference on state(s)."""
    def __init__(self, model_path: str | Path, device: str | torch.device = None, output_scale: float = 1.0):
        self.model_path = Path(model_path)
        self.device = torch.device(device) if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.policy = TMDPolicyNetwork(output_scale=output_scale).to(self.device)
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        # trained checkpoint may store dict with 'policy' key or full policy state_dict
        if isinstance(state, dict) and 'policy' in state:
            self.policy.load_state_dict(state['policy'])
        else:
            # assume file contains policy state_dict directly
            self.policy.load_state_dict(state)
        self.policy.eval()

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict control force for a single state or batch.
        state: shape (2,) or (N,2) with [displacement, velocity]
        returns: numpy array shape (1,) or (N,1)
        """
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 2:
            raise ValueError("Expected state shape (2,) or (N,2) representing [displacement, velocity]")
        with torch.no_grad():
            t = torch.from_numpy(arr).to(self.device)
            out = self.policy(t).cpu().numpy()
        return out

    def predict_scalar(self, displacement: float, velocity: float) -> float:
        """Convenience method for single scalar inputs."""
        out = self.predict(np.array([displacement, velocity]))
        return float(out.ravel()[0])

if __name__ == "__main__":
    # example usage
    MODEL_PATH = Path(__file__).parent / "tmd_rl_model.pt"  # adjust filename if needed
    inf = RLTMDInference(MODEL_PATH)
    #sample = np.array([0.1, 0.5], dtype=np.float32)  # displacement, velocity
    #  {
    #   "displacement": 0.00012358745738059112,
    #   "velocity": 0.006522938179246303,
    #   "force": -1.1536541709379569
    # },
    sample = np.array([[0.00012358745738059112, 0.006522938179246303]])
    force = inf.predict(sample)
    print(f"Predicted control force: {force.ravel()[0]:.6f}")