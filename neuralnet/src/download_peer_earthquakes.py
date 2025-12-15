"""
Download and Process PEER NGA-West2 Earthquake Data
Complete pipeline for obtaining real earthquake records for TMD training
"""

import numpy as np
import requests
from pathlib import Path
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class PEEREarthquakeDownloader:
    """
    Download and process earthquake records from PEER NGA-West2 database
    
    Note: PEER requires free account registration at https://ngawest2.berkeley.edu/
    You'll need to manually download some records, but this script helps organize them.
    """
    
    def __init__(self, data_dir: str = 'peer_earthquake_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def read_at2_file(self, filepath: str) -> Tuple[np.ndarray, float, Dict]:
        """
        Read PEER .AT2 format earthquake file
        
        Args:
            filepath: Path to .AT2 file
            
        Returns:
            acceleration: Array of ground acceleration (m/s²)
            dt: Time step (seconds)
            metadata: Dictionary with earthquake info
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header (first 4 lines typically contain metadata)
        metadata = {
            'filename': Path(filepath).name,
            'description': lines[1].strip() if len(lines) > 1 else ''
        }
        
        # Find NPTS and DT from header
        header_line = lines[3].strip()
        parts = header_line.split()
        
        npts = None
        dt = None
        for i, part in enumerate(parts):
            if 'NPTS' in part:
                npts = int(parts[i+1].replace(',', ''))
            if 'DT' in part:
                dt = float(parts[i+1])
        
        if npts is None or dt is None:
            raise ValueError(f"Could not parse NPTS and DT from {filepath}")
        
        # Read acceleration data (starts from line 4)
        data_lines = lines[4:]
        acceleration = []
        
        for line in data_lines:
            values = line.strip().split()
            acceleration.extend([float(v) for v in values])
        
        acceleration = np.array(acceleration[:npts])
        
        # Convert from g to m/s² if needed (PEER typically uses g)
        if np.max(np.abs(acceleration)) < 10:  # Likely in g units
            acceleration = acceleration * 9.81
            metadata['original_units'] = 'g'
        else:
            metadata['original_units'] = 'm/s²'
        
        metadata['npts'] = npts
        metadata['dt'] = dt
        metadata['duration'] = npts * dt
        metadata['pga'] = np.max(np.abs(acceleration))
        
        return acceleration, dt, metadata
    
    def create_synthetic_peer_like_earthquakes(self, count: int = 50) -> List[Dict]:
        """
        Generate synthetic earthquakes with characteristics similar to PEER records
        Use this while you're setting up PEER account or as supplementary data
        
        Args:
            count: Number of earthquakes to generate
            
        Returns:
            List of earthquake dictionaries
        """
        earthquakes = []
        
        # Magnitude distribution (M 4.0 to M 7.5)
        magnitudes = np.random.uniform(4.0, 7.5, count)
        
        for i, magnitude in enumerate(magnitudes):
            # Duration based on magnitude (rough empirical relationship)
            duration = 20 + (magnitude - 4.0) * 5  # 20-37 seconds
            
            # PGA based on magnitude (rough Gutenberg-Richter)
            pga_base = 0.1 * 10**(0.5 * (magnitude - 5.0))  # in g
            pga = pga_base * np.random.uniform(0.5, 2.0)  # Add variability
            pga = np.clip(pga, 0.05, 1.2)  # Reasonable bounds
            
            # Sampling rate
            dt = 0.01  # 100 Hz initially, will resample to 50 Hz
            t = np.arange(0, duration, dt)
            
            # Generate realistic earthquake signal
            accel = self._generate_realistic_earthquake(t, pga, magnitude)
            
            # Resample to 50 Hz (dt = 0.02s) to match your system
            accel_50hz = accel[::2]  # Downsample by factor of 2
            t_50hz = t[::2]
            
            earthquake = {
                'id': f'synthetic_peer_{i+1:03d}',
                'magnitude': float(magnitude),
                'pga_g': float(pga),
                'pga_ms2': float(pga * 9.81),
                'duration': float(duration),
                'dt': 0.02,
                'npts': len(accel_50hz),
                'acceleration': accel_50hz.tolist(),
                'time': t_50hz.tolist(),
                'source': 'synthetic_peer_like'
            }
            
            earthquakes.append(earthquake)
        
        return earthquakes
    
    def _generate_realistic_earthquake(self, t: np.ndarray, pga: float, magnitude: float) -> np.ndarray:
        """
        Generate realistic earthquake acceleration time history
        Uses multiple frequency components and envelope function
        """
        # Dominant frequencies based on magnitude
        if magnitude < 5.0:
            freqs = [2.0, 3.5, 5.0, 7.0]  # Higher frequencies for small EQ
            weights = [0.4, 0.3, 0.2, 0.1]
        elif magnitude < 6.5:
            freqs = [1.0, 2.0, 3.0, 4.5]  # Mid frequencies
            weights = [0.35, 0.35, 0.2, 0.1]
        else:
            freqs = [0.5, 1.0, 1.5, 2.5]  # Lower frequencies for large EQ
            weights = [0.4, 0.3, 0.2, 0.1]
        
        # Generate multi-frequency signal
        accel = np.zeros_like(t)
        for freq, weight in zip(freqs, weights):
            phase = np.random.uniform(0, 2*np.pi)
            accel += weight * np.sin(2 * np.pi * freq * t + phase)
        
        # Add noise
        accel += 0.1 * np.random.randn(len(t))
        
        # Apply envelope (buildup, strong shaking, decay)
        duration = t[-1]
        envelope = self._earthquake_envelope(t, duration)
        accel = accel * envelope
        
        # Scale to desired PGA
        current_pga = np.max(np.abs(accel))
        accel = accel * (pga * 9.81 / current_pga)  # Scale to m/s²
        
        return accel
    
    def _earthquake_envelope(self, t: np.ndarray, duration: float) -> np.ndarray:
        """
        Realistic earthquake envelope function
        Slow buildup, strong shaking, exponential decay
        """
        t1 = duration * 0.15  # Buildup phase (15% of duration)
        t2 = duration * 0.50  # Strong shaking (35% of duration)
        
        envelope = np.zeros_like(t)
        
        # Buildup
        mask1 = t < t1
        envelope[mask1] = (t[mask1] / t1) ** 2
        
        # Strong shaking
        mask2 = (t >= t1) & (t < t2)
        envelope[mask2] = 1.0
        
        # Exponential decay
        mask3 = t >= t2
        decay_time = t[mask3] - t2
        envelope[mask3] = np.exp(-decay_time / (duration * 0.15))
        
        return envelope
    
    def save_earthquake_dataset(self, earthquakes: List[Dict], filename: str = 'peer_earthquake_dataset.json'):
        """Save earthquake dataset to JSON"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(earthquakes, f, indent=2)
        
        print(f"✅ Saved {len(earthquakes)} earthquakes to {filepath}")
        return filepath
    
    def load_earthquake_dataset(self, filename: str = 'peer_earthquake_dataset.json') -> List[Dict]:
        """Load earthquake dataset from JSON"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'r') as f:
            earthquakes = json.load(f)
        
        print(f"✅ Loaded {len(earthquakes)} earthquakes from {filepath}")
        return earthquakes
    
    def visualize_earthquake(self, earthquake: Dict, save_path: str = None):
        """Visualize earthquake acceleration time history"""
        t = np.array(earthquake['time'])
        accel = np.array(earthquake['acceleration'])
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time history
        axes[0].plot(t, accel, 'b-', linewidth=0.5)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].set_title(f"Earthquake {earthquake['id']} - M{earthquake['magnitude']:.1f}, PGA={earthquake['pga_g']:.3f}g")
        axes[0].grid(True, alpha=0.3)
        
        # Frequency content
        from scipy import signal as sp_signal
        f, Pxx = sp_signal.welch(accel, fs=1/earthquake['dt'], nperseg=256)
        axes[1].semilogy(f, Pxx, 'r-', linewidth=1)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power Spectral Density')
        axes[1].set_title('Frequency Content')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 20])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_earthquake_statistics(self, earthquakes: List[Dict]) -> Dict:
        """Calculate statistics about the earthquake dataset"""
        magnitudes = [eq['magnitude'] for eq in earthquakes]
        pgas = [eq['pga_g'] for eq in earthquakes]
        durations = [eq['duration'] for eq in earthquakes]
        
        stats = {
            'count': len(earthquakes),
            'magnitude': {
                'min': min(magnitudes),
                'max': max(magnitudes),
                'mean': np.mean(magnitudes),
                'std': np.std(magnitudes)
            },
            'pga_g': {
                'min': min(pgas),
                'max': max(pgas),
                'mean': np.mean(pgas),
                'std': np.std(pgas)
            },
            'duration': {
                'min': min(durations),
                'max': max(durations),
                'mean': np.mean(durations),
                'std': np.std(durations)
            }
        }
        
        return stats


def main():
    """Main execution"""
    print("="*70)
    print("PEER NGA-WEST2 EARTHQUAKE DATA PREPARATION")
    print("="*70)
    print()
    
    downloader = PEEREarthquakeDownloader()
    
    # Option 1: If you have real PEER .AT2 files
    # Uncomment and modify this section when you have real files
    """
    print("Reading real PEER .AT2 files...")
    peer_files = [
        'path/to/RSN6_IMPVALL.I_I-ELC180.AT2',  # El Centro
        'path/to/RSN952_NORTHR_MUL009.AT2',     # Northridge
        'path/to/RSN1111_KOBE_KBU000.AT2',      # Kobe
        # Add more files...
    ]
    
    real_earthquakes = []
    for filepath in peer_files:
        try:
            accel, dt, metadata = downloader.read_at2_file(filepath)
            earthquake = {
                'id': metadata['filename'].replace('.AT2', ''),
                'magnitude': None,  # Add manually if known
                'pga_g': metadata['pga'] / 9.81,
                'pga_ms2': metadata['pga'],
                'duration': metadata['duration'],
                'dt': dt,
                'npts': metadata['npts'],
                'acceleration': accel.tolist(),
                'time': (np.arange(metadata['npts']) * dt).tolist(),
                'source': 'PEER_NGA_West2'
            }
            real_earthquakes.append(earthquake)
            print(f"  ✅ Loaded {earthquake['id']}")
        except Exception as e:
            print(f"  ❌ Error loading {filepath}: {e}")
    """
    
    # Option 2: Generate synthetic PEER-like earthquakes
    print("Generating 50 synthetic PEER-like earthquakes...")
    synthetic_earthquakes = downloader.create_synthetic_peer_like_earthquakes(count=50)
    print(f"  ✅ Generated {len(synthetic_earthquakes)} earthquakes")
    print()
    
    # Combine real and synthetic (if you have real ones)
    all_earthquakes = synthetic_earthquakes  # + real_earthquakes
    
    # Save dataset
    dataset_path = downloader.save_earthquake_dataset(all_earthquakes)
    print()
    
    # Show statistics
    stats = downloader.get_earthquake_statistics(all_earthquakes)
    print("Dataset Statistics:")
    print(f"  Total earthquakes: {stats['count']}")
    print(f"  Magnitude range: {stats['magnitude']['min']:.1f} - {stats['magnitude']['max']:.1f}")
    print(f"  PGA range: {stats['pga_g']['min']:.3f}g - {stats['pga_g']['max']:.3f}g")
    print(f"  Duration range: {stats['duration']['min']:.1f}s - {stats['duration']['max']:.1f}s")
    print()
    
    # Visualize a few examples
    print("Visualizing sample earthquakes...")
    for i in [0, 10, 25, 49]:  # Sample indices
        downloader.visualize_earthquake(
            all_earthquakes[i],
            save_path=f'earthquake_sample_{i+1}.png'
        )
    print()
    
    print("="*70)
    print("✅ EARTHQUAKE DATA READY FOR TRAINING")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Run: python generate_training_data_from_peer.py")
    print("2. This will simulate building responses to these earthquakes")
    print("3. Then run: python train_neural_network.py")
    print()
    print(f"Dataset saved to: {dataset_path}")


if __name__ == '__main__':
    main()
