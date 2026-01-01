"""
Calculate optimal TMD parameters - CORRECTED
"""
import numpy as np
import math

# Building parameters
n_floors = 12
floor_mass = 2.0e5
k_typical = 2.0e7

# Build mass and stiffness matrices
M_floors = floor_mass * np.ones(n_floors)
K = np.zeros((n_floors, n_floors))

k = k_typical * np.ones(n_floors)
k[7] = 0.60 * k_typical

for i in range(n_floors):
    if i == 0:
        K[i,i] = k[i]
    else:
        K[i,i] = k[i] + k[i-1]
        K[i,i-1] = -k[i-1]
        K[i-1,i] = -k[i-1]

M_mat = np.diag(M_floors)

eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M_mat) @ K)
natural_frequencies = np.sqrt(eigenvalues.real) / (2 * np.pi)
sorted_indices = np.argsort(natural_frequencies)
natural_frequencies = natural_frequencies[sorted_indices]

f_1 = natural_frequencies[0]
omega_1 = f_1 * 2 * np.pi

print("Building fundamental frequency: {:.3f} Hz".format(f_1))

# TMD
tmd_mass = 0.02 * floor_mass
total_building_mass = n_floors * floor_mass
mu = tmd_mass / total_building_mass

print("TMD mass: {:.0f} kg".format(tmd_mass))
print("Total building mass: {:.0f} kg".format(total_building_mass))
print("Mass ratio μ: {:.4f}".format(mu))

# Den Hartog optimal formulas
f_opt = 1 / (1 + mu)
omega_tmd_opt = f_opt * omega_1
f_tmd_opt = omega_tmd_opt / (2 * np.pi)

zeta_opt = math.sqrt(3 * mu / (8 * (1 + mu)))

k_tmd_opt = tmd_mass * (omega_tmd_opt ** 2)
c_tmd_opt = 2 * zeta_opt * math.sqrt(k_tmd_opt * tmd_mass)

print("\n" + "="*70)
print("OPTIMAL TMD (Den Hartog)")
print("="*70)
print("Frequency ratio: {:.4f}".format(f_opt))
print("TMD frequency: {:.3f} Hz".format(f_tmd_opt))
print("TMD stiffness: {:.0f} N/m".format(k_tmd_opt))
print("TMD damping: {:.0f} N·s/m".format(c_tmd_opt))
print("Damping ratio: {:.4f} ({:.2f}%)".format(zeta_opt, zeta_opt*100))

# Current wrong
k_wrong = 50000
c_wrong = 2000
omega_wrong = math.sqrt(k_wrong / tmd_mass)
f_wrong = omega_wrong / (2 * np.pi)
zeta_wrong = c_wrong / (2 * math.sqrt(k_wrong * tmd_mass))

print("\n" + "="*70)
print("CURRENT (WRONG) TMD")
print("="*70)
print("TMD frequency: {:.3f} Hz (ratio: {:.3f})".format(f_wrong, f_wrong/f_1))
print("TMD stiffness: {:.0f} N/m".format(k_wrong))
print("TMD damping: {:.0f} N·s/m".format(c_wrong))
print("Damping ratio: {:.4f} ({:.2f}%)".format(zeta_wrong, zeta_wrong*100))

print("\n" + "="*70)
print("PROBLEM")
print("="*70)
print("Current TMD is tuned to {:.1f}x the building frequency!".format(f_wrong/f_1))
print("Optimal would be {:.2f}x (nearly 1:1 tuning)".format(f_opt))
print("\nCurrent TMD is {:.1f}x TOO STIFF (needs to be SOFTER)".format(k_wrong/k_tmd_opt))
print("Current TMD is {:.1f}x OVER-DAMPED (needs LESS damping)".format(c_wrong/c_tmd_opt))
