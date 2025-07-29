#!/usr/bin/env python3
"""
Create synthetic files for each WOSIS depth range
- real.csv: main dataset 
- test.csv: exactly 10% size of real.csv
Copy these to synthetic folders within each WOSIS directory
"""

import pandas as pd
import os
from pathlib import Path
import shutil

def create_synthetic_files():
    """Create synthetic folders with real.csv and test.csv files"""
    
    data_dir = Path('/root/WOSIS_enhancement/data')
    wosis_folders = ['WOSIS_0_15cm', 'WOSIS_15_30cm', 'WOSIS_30_60cm', 'WOSIS_60_100cm']
    
    for folder_name in wosis_folders:
        folder_path = data_dir / folder_name
        print(f"\nProcessing {folder_name}...")
        
        # Read the main CSV file
        main_csv_path = folder_path / f"{folder_name}.csv"
        if not main_csv_path.exists():
            print(f"Main CSV file not found: {main_csv_path}")
            continue
            
        print(f"Reading {main_csv_path}")
        df_main = pd.read_csv(main_csv_path)
        print(f"Main dataset size: {len(df_main)} records")
        
        # Create test dataset that is exactly 10% of main dataset
        test_size = int(len(df_main) * 0.1)
        print(f"Creating test dataset with {test_size} records (10% of main)")
        
        # Sample exactly 10% for test set
        df_test = df_main.sample(n=test_size, random_state=42)
        
        # Create synthetic folder
        synthetic_dir = folder_path / 'synthetic'
        synthetic_dir.mkdir(exist_ok=True)
        print(f"Created synthetic directory: {synthetic_dir}")
        
        # Save real.csv (main dataset)
        real_csv_path = synthetic_dir / 'real.csv'
        df_main.to_csv(real_csv_path, index=False)
        print(f"Saved real.csv: {len(df_main)} records -> {real_csv_path}")
        
        # Save test.csv (10% subset)
        test_csv_path = synthetic_dir / 'test.csv'
        df_test.to_csv(test_csv_path, index=False)
        print(f"Saved test.csv: {len(df_test)} records -> {test_csv_path}")
        
        # Verify the ratio
        ratio = len(df_test) / len(df_main)
        print(f"Test/Real ratio: {ratio:.3f} (target: 0.100)")
        
        # Show some statistics
        print(f"Date distribution in {folder_name}:")
        real_dates = df_main['date'].value_counts()
        test_dates = df_test['date'].value_counts()
        real_known = len(df_main[df_main['date'] != 'Unknown'])
        test_known = len(df_test[df_test['date'] != 'Unknown'])
        print(f"  Real: {real_known}/{len(df_main)} with dates ({real_known/len(df_main)*100:.1f}%)")
        print(f"  Test: {test_known}/{len(df_test)} with dates ({test_known/len(df_test)*100:.1f}%)")

def main():
    print("Creating synthetic files for WOSIS datasets...")
    print("="*60)
    
    create_synthetic_files()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Print summary of created files
    data_dir = Path('/root/lucica/WOSIS_enhancement/data')
    wosis_folders = ['WOSIS_0_15cm', 'WOSIS_15_30cm', 'WOSIS_30_60cm', 'WOSIS_60_100cm']
    
    total_real_records = 0
    total_test_records = 0
    
    for folder_name in wosis_folders:
        synthetic_dir = data_dir / folder_name / 'synthetic'
        real_path = synthetic_dir / 'real.csv'
        test_path = synthetic_dir / 'test.csv'
        
        if real_path.exists() and test_path.exists():
            real_size = len(pd.read_csv(real_path))
            test_size = len(pd.read_csv(test_path))
            total_real_records += real_size
            total_test_records += test_size
            
            print(f"{folder_name}:")
            print(f"  real.csv: {real_size:,} records")
            print(f"  test.csv: {test_size:,} records ({test_size/real_size*100:.1f}%)")
            print(f"  Location: {synthetic_dir}")
    
    print(f"\nTotals:")
    print(f"  All real.csv files: {total_real_records:,} records")
    print(f"  All test.csv files: {total_test_records:,} records")
    print(f"  Overall test ratio: {total_test_records/total_real_records*100:.1f}%")
    
    print("\nSynthetic file creation completed successfully!")

if __name__ == "__main__":
    main() 