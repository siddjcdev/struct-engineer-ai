#!/usr/bin/env python3
"""
TMD Simulation Results Visualization
Generates comprehensive plots for science fair presentation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class TMDVisualizer:
    def __init__(self):
        """Initialize with test case data from the simulation results"""
        self.test_cases = {
            'Test 1': {
                'name': 'Stationary Wind\n+ Earthquake',
                'loading': '12 m/s + 0.35g',
                'baseline_dcr': 1.481,
                'tmd_dcr': 1.220,
                'dcr_reduction': 17.6,
                'drift_reduction': 16.1,
                'roof_reduction': -7.0,
                'floor': 9,
                'mass_ratio': 0.060,
                'damping': 0.050
            },
            'Test 2': {
                'name': 'Turbulent Wind\n+ Earthquake',
                'loading': '25 m/s + 0.35g',
                'baseline_dcr': 1.488,
                'tmd_dcr': 1.381,
                'dcr_reduction': 7.2,
                'drift_reduction': 2.0,
                'roof_reduction': 0.3,
                'floor': 2,
                'mass_ratio': 0.300,
                'damping': 0.050
            },
            'Test 3': {
                'name': 'Small Earthquake\n(M 4.5)',
                'loading': '0.10g',
                'baseline_dcr': 1.426,
                'tmd_dcr': 1.394,
                'dcr_reduction': 2.2,
                'drift_reduction': 2.7,
                'roof_reduction': -1.7,
                'floor': 5,
                'mass_ratio': 0.250,
                'damping': 0.090
            },
            'Test 4': {
                'name': 'Large Earthquake\n(M 6.9)',
                'loading': '0.40g',
                'baseline_dcr': 1.597,
                'tmd_dcr': 1.559,
                'dcr_reduction': 2.4,
                'drift_reduction': 0.4,
                'roof_reduction': -0.2,
                'floor': 8,
                'mass_ratio': 0.100,
                'damping': 0.050
            },
            'Test 5': {
                'name': 'Extreme Combined\n(Hurricane + Earthquake)',
                'loading': '50 m/s + 0.40g',
                'baseline_dcr': 1.585,
                'tmd_dcr': 1.582,
                'dcr_reduction': 0.2,
                'drift_reduction': -0.8,
                'roof_reduction': -0.6,
                'floor': 12,
                'mass_ratio': 0.240,
                'damping': 0.050
            },
            'Test 6': {
                'name': 'Noisy Data\n(10% noise)',
                'loading': '0.39g + noise',
                'baseline_dcr': 1.583,
                'tmd_dcr': 1.552,
                'dcr_reduction': 1.9,
                'drift_reduction': 0.0,
                'roof_reduction': -0.2,
                'floor': 8,
                'mass_ratio': 0.110,
                'damping': 0.050
            }
        }
    
    def plot_dcr_comparison(self, save_path='dcr_comparison.png'):
        """Plot DCR reduction comparison across all test cases"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        tests = list(self.test_cases.keys())
        dcr_reductions = [self.test_cases[t]['dcr_reduction'] for t in tests]
        names = [self.test_cases[t]['name'] for t in tests]
        
        # Color code by performance
        colors = ['#2ecc71' if x > 10 else '#f39c12' if x > 5 else '#e74c3c' for x in dcr_reductions]
        
        bars = ax.bar(range(len(tests)), dcr_reductions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, dcr_reductions)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Test Case', fontsize=14, fontweight='bold')
        ax.set_ylabel('DCR Reduction (%)', fontsize=14, fontweight='bold')
        ax.set_title('TMD Effectiveness: DCR Reduction Across Loading Scenarios', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(tests)))
        ax.set_xticklabels(names, fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Excellent (>10%)'),
            Patch(facecolor='#f39c12', label='Moderate (5-10%)'),
            Patch(facecolor='#e74c3c', label='Limited (<5%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_performance_tradeoffs(self, save_path='performance_tradeoffs.png'):
        """Plot drift reduction vs. roof displacement change"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        tests = list(self.test_cases.keys())
        drift_reductions = [self.test_cases[t]['drift_reduction'] for t in tests]
        roof_reductions = [self.test_cases[t]['roof_reduction'] for t in tests]
        dcr_reductions = [self.test_cases[t]['dcr_reduction'] for t in tests]
        
        # Size points by DCR reduction
        sizes = [100 + 50*abs(x) for x in dcr_reductions]
        
        scatter = ax.scatter(drift_reductions, roof_reductions, s=sizes, 
                           c=dcr_reductions, cmap='RdYlGn', alpha=0.7, 
                           edgecolors='black', linewidth=2)
        
        # Add test labels
        for i, test in enumerate(tests):
            ax.annotate(test, (drift_reductions[i], roof_reductions[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add quadrant lines
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Annotate quadrants
        ax.text(12, 5, 'Win-Win\n(Both reduce)', fontsize=11, ha='center', 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.text(12, -5, 'Trade-off\n(Drift↓, Roof↑)', fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        ax.set_xlabel('Drift Reduction (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Roof Displacement Reduction (%)', fontsize=14, fontweight='bold')
        ax.set_title('Performance Trade-offs: Drift vs. Roof Displacement', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('DCR Reduction (%)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_tmd_parameters(self, save_path='tmd_parameters.png'):
        """Plot TMD mass ratio vs. DCR reduction"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        tests = list(self.test_cases.keys())
        mass_ratios = [self.test_cases[t]['mass_ratio'] * 100 for t in tests]
        dcr_reductions = [self.test_cases[t]['dcr_reduction'] for t in tests]
        floors = [self.test_cases[t]['floor'] for t in tests]
        
        # Plot 1: Mass Ratio vs. DCR Reduction
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(tests)))
        ax1.scatter(mass_ratios, dcr_reductions, s=200, c=colors1, alpha=0.7, 
                   edgecolors='black', linewidth=2)
        
        for i, test in enumerate(tests):
            ax1.annotate(test, (mass_ratios[i], dcr_reductions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax1.set_xlabel('TMD Mass Ratio (% of building mass)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('DCR Reduction (%)', fontsize=13, fontweight='bold')
        ax1.set_title('TMD Mass Ratio vs. Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Floor Location vs. DCR Reduction
        ax2.scatter(floors, dcr_reductions, s=200, c=colors1, alpha=0.7, 
                   edgecolors='black', linewidth=2)
        
        for i, test in enumerate(tests):
            ax2.annotate(test, (floors[i], dcr_reductions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('TMD Floor Location', fontsize=13, fontweight='bold')
        ax2.set_ylabel('DCR Reduction (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Optimal TMD Placement', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(2, 13))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_loading_intensity_correlation(self, save_path='loading_intensity.png'):
        """Plot loading intensity vs. TMD effectiveness"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Manually categorize loading intensity
        loading_categories = {
            'Test 1': 1.5,  # Moderate
            'Test 2': 2.5,  # High
            'Test 3': 0.5,  # Low
            'Test 4': 3.5,  # Very High
            'Test 5': 5.0,  # Extreme
            'Test 6': 3.5   # Very High (with noise)
        }
        
        tests = list(self.test_cases.keys())
        intensities = [loading_categories[t] for t in tests]
        dcr_reductions = [self.test_cases[t]['dcr_reduction'] for t in tests]
        
        colors = ['#2ecc71' if x > 10 else '#f39c12' if x > 5 else '#e74c3c' for x in dcr_reductions]
        
        ax.scatter(intensities, dcr_reductions, s=300, c=colors, alpha=0.7, 
                  edgecolors='black', linewidth=2)
        
        for i, test in enumerate(tests):
            ax.annotate(test, (intensities[i], dcr_reductions[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=11)
        
        # Fit exponential decay curve
        from scipy.optimize import curve_fit
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_smooth = np.linspace(min(intensities), max(intensities), 100)
        try:
            popt, _ = curve_fit(exp_decay, intensities, dcr_reductions, p0=[20, 0.5, 0])
            y_smooth = exp_decay(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, 'r--', linewidth=2, alpha=0.6, 
                   label='Exponential Decay Fit')
        except:
            pass
        
        ax.set_xlabel('Loading Intensity (Arbitrary Units)', fontsize=14, fontweight='bold')
        ax.set_ylabel('DCR Reduction (%)', fontsize=14, fontweight='bold')
        ax.set_title('TMD Effectiveness vs. Loading Intensity', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add intensity labels
        ax.text(0.5, -2, 'Low', fontsize=10, ha='center')
        ax.text(1.5, -2, 'Moderate', fontsize=10, ha='center')
        ax.text(2.5, -2, 'High', fontsize=10, ha='center')
        ax.text(3.5, -2, 'Very High', fontsize=10, ha='center')
        ax.text(5.0, -2, 'Extreme', fontsize=10, ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_summary_table(self, save_path='summary_table.png'):
        """Create a comprehensive summary table"""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        data = []
        for test, values in self.test_cases.items():
            data.append([
                test,
                values['loading'],
                f"{values['baseline_dcr']:.3f}",
                f"{values['tmd_dcr']:.3f}",
                f"{values['dcr_reduction']:.1f}%",
                f"{values['drift_reduction']:.1f}%",
                f"Floor {values['floor']}",
                f"{values['mass_ratio']*100:.1f}%"
            ])
        
        columns = ['Test', 'Loading', 'Baseline\nDCR', 'TMD\nDCR', 
                  'DCR\nReduction', 'Drift\nReduction', 'TMD\nLocation', 'Mass\nRatio']
        
        table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        
        # Color code performance
        for i in range(1, len(data) + 1):
            dcr_reduction = float(data[i-1][4].strip('%'))
            if dcr_reduction > 10:
                color = '#d5f4e6'
            elif dcr_reduction > 5:
                color = '#fff4e6'
            else:
                color = '#ffe6e6'
            
            for j in range(len(columns)):
                table[(i, j)].set_facecolor(color)
        
        # Header styling
        for j in range(len(columns)):
            table[(0, j)].set_facecolor('#3498db')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        plt.title('TMD Performance Summary: All Test Cases', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_all_plots(self, output_dir='visualizations'):
        """Generate all visualization plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING TMD VISUALIZATION PLOTS")
        print("="*60 + "\n")
        
        self.plot_dcr_comparison(f'{output_dir}/dcr_comparison.png')
        self.plot_performance_tradeoffs(f'{output_dir}/performance_tradeoffs.png')
        self.plot_tmd_parameters(f'{output_dir}/tmd_parameters.png')
        self.plot_loading_intensity_correlation(f'{output_dir}/loading_intensity.png')
        self.plot_summary_table(f'{output_dir}/summary_table.png')
        
        print("\n" + "="*60)
        print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print(f"✓ Output directory: {output_dir}/")
        print("="*60 + "\n")

if __name__ == "__main__":
    visualizer = TMDVisualizer()
    visualizer.generate_all_plots()
