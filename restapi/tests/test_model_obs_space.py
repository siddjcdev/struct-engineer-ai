"""Quick test to check model observation space requirements"""
from stable_baselines3 import SAC
from pathlib import Path

MODEL1 = Path("../models/rl_cl_final_robust.zip")
MODEL2 = Path("../models/rl_cl_dcr_train_5_final.zip")

print("Loading models...")
model1 = SAC.load(MODEL1)
model2 = SAC.load(MODEL2)

print(f"\nModel 1 (rl_cl_final_robust):")
print(f"  Observation space: {model1.observation_space}")
print(f"  Observation shape: {model1.observation_space.shape}")

print(f"\nModel 2 (rl_cl_dcr_train_5_final):")
print(f"  Observation space: {model2.observation_space}")
print(f"  Observation shape: {model2.observation_space.shape}")
