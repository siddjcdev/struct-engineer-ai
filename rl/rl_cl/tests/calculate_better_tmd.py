"""
Calculate TMD parameters using optimal frequency ratio of 0.80
"""
import math

building_freq = 0.193  # Hz
tmd_mass = 4000.0
total_mass = 12 * 200000

# Best frequency ratio from sweep
ratio = 0.80
f_tmd = ratio * building_freq
omega_tmd = f_tmd * 2 * math.pi

# Stiffness
k_tmd = tmd_mass * (omega_tmd ** 2)

# Damping (Den Hartog optimal)
mu = tmd_mass / total_mass
zeta = math.sqrt(3 * mu / (8 * (1 + mu)))
c_tmd = 2 * zeta * math.sqrt(k_tmd * tmd_mass)

print("="*70)
print("OPTIMIZED TMD PARAMETERS (from parameter sweep)")
print("="*70)
print(f"Frequency ratio: {ratio:.2f}")
print(f"TMD frequency: {f_tmd:.3f} Hz")
print(f"TMD stiffness: {k_tmd:.0f} N/m")
print(f"TMD damping: {c_tmd:.0f} N·s/m")
print(f"Damping ratio: {zeta:.4f} ({zeta*100:.2f}%)")
print()
print(f"Expected performance:")
print(f"  No TMD: 20.90 cm")
print(f"  Passive TMD: 20.16 cm")
print(f"  Reduction: {((20.90-20.16)/20.90*100):.1f}%")
print("="*70)
