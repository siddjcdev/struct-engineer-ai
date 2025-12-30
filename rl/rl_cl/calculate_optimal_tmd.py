"""
Calculate optimal TMD parameters for the building
"""
import numpy as np
import math

# Building parameters (from tmd_environment.py)
n_floors = 12
floor_mass = 2.0e5  # 200,000 kg
k_typical = 2.0e7   # 20 MN/m

# Build mass and stiffness matrices
M_floors = floor_mass * np.ones(n_floors)
K = np.zeros((n_floors, n_floors))

# Story stiffness
k = k_typical * np.ones(n_floors)
k[7] = 0.60 * k_typical  # Soft story

# Build stiffness matrix
for i in range(n_floors):
    if i == 0:
        K[i,i] = k[i]
    else:
        K[i,i] = k[i] + k[i-1]
        K[i,i-1] = -k[i-1]
        K[i-1,i] = -k[i-1]

# Build mass matrix
M_mat = np.diag(M_floors)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M_mat) @ K)
natural_frequencies = np.sqrt(eigenvalues.real) / (2 * np.pi)
sorted_indices = np.argsort(natural_frequencies)
natural_frequencies = natural_frequencies[sorted_indices]

print("=" * 70)
print("BUILDING NATURAL FREQUENCIES")
print("=" * 70)
for i, freq in enumerate(natural_frequencies[:5]):
    print(f"Mode {i+1}: {freq:.3f} Hz")

# Fundamental frequency
omega_1 = natural_frequencies[0] * 2 * np.pi  # rad/s
f_1 = natural_frequencies[0]  # Hz

print(f"\nFundamental frequency: {f_1:.3f} Hz ({omega_1:.3f} rad/s)")

# TMD parameters
tmd_mass = 0.02 * floor_mass  # 2% mass ratio = 4000 kg
mu = tmd_mass / (n_floors * floor_mass)  # Mass ratio relative to total building

print("\n" + "=" * 70)
print("OPTIMAL TMD PARAMETERS (Den Hartog's formulas)")
print("=" * 70)

# Den Hartog's optimal tuning formulas for undamped structure
# Optimal frequency ratio
f_opt = 1 / (1 + mu)
omega_tmd = f_opt * omega_1

# Optimal damping ratio
zeta_opt = math.sqrt(3 * mu / (8 * (1 + mu)))

# TMD stiffness
k_tmd = tmd_mass * (omega_tmd ** 2)

# TMD damping coefficient
c_tmd = 2 * zeta_opt * math.sqrt(k_tmd * tmd_mass)

print(f"Mass ratio (μ): {mu:.4f} ({mu*100:.2f}%)")
print(f"Optimal frequency ratio: {f_opt:.4f}")
print(f"TMD natural frequency: {omega_tmd/(2*np.pi):.3f} Hz")
print(f"Optimal damping ratio (ζ): {zeta_opt:.4f} ({zeta_opt*100:.2f}%)")
print(f"TMD stiffness (k_tmd): {k_tmd:.0f} N/m ({k_tmd/1e6:.2f} MN/m)")
print(f"TMD damping (c_tmd): {c_tmd:.0f} N·s/m")

# Current wrong values
print("\n" + "=" * 70)
print("CURRENT (WRONG) TMD PARAMETERS")
print("=" * 70)
k_tmd_wrong = 50e3  # 50 kN/m
c_tmd_wrong = 2000  # 2000 N·s/m
omega_tmd_wrong = math.sqrt(k_tmd_wrong / tmd_mass)
f_tmd_wrong = omega_tmd_wrong / (2 * np.pi)
zeta_tmd_wrong = c_tmd_wrong / (2 * math.sqrt(k_tmd_wrong * tmd_mass))

print(f"TMD stiffness: {k_tmd_wrong:.0f} N/m ({k_tmd_wrong/1e6:.4f} MN/m)")
print(f"TMD damping: {c_tmd_wrong:.0f} N·s/m")
print(f"TMD frequency: {f_tmd_wrong:.3f} Hz (tuning ratio: {f_tmd_wrong/f_1:.3f})")
print(f"TMD damping ratio: {zeta_tmd_wrong:.4f} ({zeta_tmd_wrong*100:.2f}%)")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Frequency ratio: {f_tmd_wrong/f_1:.3f} (current) vs {f_opt:.3f} (optimal)")
print(f"Stiffness: {k_tmd_wrong/1e6:.4f} MN/m (current) vs {k_tmd/1e6:.2f} MN/m (optimal)")
print(f"Damping ratio: {zeta_tmd_wrong*100:.1f}% (current) vs {zeta_opt*100:.1f}% (optimal)")
print(f"\n⚠️  Current TMD is {k_tmd/k_tmd_wrong:.1f}x TOO SOFT!")
print(f"⚠️  Current TMD is {c_tmd/c_tmd_wrong:.1f}x UNDER-DAMPED!")
