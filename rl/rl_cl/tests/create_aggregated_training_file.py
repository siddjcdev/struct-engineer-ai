"""
CREATE AGGREGATED TRAINING FILE
================================

Takes 80% of data from all earthquake CSV files and combines them into
a single aggregated training file.

Usage: python create_aggregated_training_file.py --earthquakes <files> --output <output_file>
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path


def create_aggregated_training_file(earthquake_files, output_file, train_split=0.8):
    """
    Aggregate 80% of data from all earthquake files into a single training file.

    Args:
        earthquake_files: List of earthquake CSV file paths
        output_file: Path to save the aggregated training file
        train_split: Fraction of data to use (default 0.8 for 80%)
    """

    print("="*70)
    print("  CREATING AGGREGATED TRAINING FILE")
    print("="*70)
    print(f"\n📂 Processing {len(earthquake_files)} earthquake files...")
    print(f"   Train split: {train_split*100:.0f}%\n")

    all_data = []
    file_stats = []

    for eq_file in earthquake_files:
        if not os.path.exists(eq_file):
            print(f"⚠️  Skipping {eq_file} (not found)")
            continue

        # Read the earthquake file
        df = pd.read_csv(eq_file, header=None, names=['time', 'acceleration'])
        total_rows = len(df)

        # Take first 80% of the data
        train_rows = int(total_rows * train_split)
        train_data = df.iloc[:train_rows]

        # Add to aggregated data
        all_data.append(train_data)

        file_name = os.path.basename(eq_file)
        file_stats.append({
            'file': file_name,
            'total_rows': total_rows,
            'train_rows': train_rows,
            'duration': df['time'].max()
        })

        print(f"   ✓ {file_name}")
        print(f"      Total: {total_rows} rows, Using: {train_rows} rows ({train_split*100:.0f}%)")

    if not all_data:
        print("\n❌ No valid earthquake files found!")
        return

    # Combine all data
    print(f"\n🔄 Combining data from {len(all_data)} files...")
    aggregated_df = pd.concat(all_data, ignore_index=True)

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save aggregated file
    aggregated_df.to_csv(output_file, index=False, header=False)

    # Summary statistics
    total_original_rows = sum(stat['total_rows'] for stat in file_stats)
    total_train_rows = sum(stat['train_rows'] for stat in file_stats)
    final_rows = len(aggregated_df)

    print("\n" + "="*70)
    print("  ✅ AGGREGATION COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Files processed: {len(file_stats)}")
    print(f"   Total original rows: {total_original_rows:,}")
    print(f"   Total train rows ({train_split*100:.0f}%): {total_train_rows:,}")
    print(f"   Aggregated file rows: {final_rows:,}")
    print(f"\n💾 Saved to: {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("\n🚀 Use this file for training:")
    print(f"   python train_rl_cl.py --earthquakes {output_file}")
    print("="*70 + "\n")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create aggregated training file from multiple earthquake files'
    )
    parser.add_argument('--earthquakes', nargs='+', required=True,
                       help='Earthquake CSV files to aggregate')
    parser.add_argument('--output', default='matlab/datasets/aggregated_train_80pct.csv',
                       help='Output file path (default: matlab/datasets/aggregated_train_80pct.csv)')
    parser.add_argument('--split', type=float, default=0.8,
                       help='Training split fraction (default: 0.8 for 80%%)')

    args = parser.parse_args()

    print("\n🚀 Starting aggregation process...\n")
    create_aggregated_training_file(args.earthquakes, args.output, args.split)
    print("\n✅ All done!")
