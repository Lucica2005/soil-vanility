#!/usr/bin/env python3
"""
Split wosis_large.csv into four depth-based files
This script categorizes data by depth ranges and replaces existing WOSIS CSV files
"""

import pandas as pd
import os
import shutil
from pathlib import Path

def categorize_by_depth(df):
    """
    Categorize data by depth ranges based on overlap with target depth ranges
    
    Target ranges:
    - 0-15cm
    - 15-30cm  
    - 30-60cm
    - 60-100cm
    """
    
    # Define target depth ranges
    depth_ranges = {
        'WOSIS_0_15cm': (0, 15),
        'WOSIS_15_30cm': (15, 30),
        'WOSIS_30_60cm': (30, 60), 
        'WOSIS_60_100cm': (60, 100)
    }
    
    categorized_data = {}
    
    for category, (target_min, target_max) in depth_ranges.items():
        print(f"\nProcessing category: {category} (depth range: {target_min}-{target_max}cm)")
        
        # Find records that overlap with the target depth range
        # A record overlaps if: upper_depth < target_max AND lower_depth > target_min
        mask = (df['upper_depth'] < target_max) & (df['lower_depth'] > target_min)
        
        # Additional filtering: prefer records that are primarily within the target range
        # Calculate overlap percentage and prioritize records with higher overlap
        df_subset = df[mask].copy()
        
        if len(df_subset) > 0:
            # Calculate overlap with target range for each record
            df_subset['overlap_start'] = df_subset[['upper_depth']].apply(
                lambda x: max(x['upper_depth'], target_min), axis=1)
            df_subset['overlap_end'] = df_subset[['lower_depth']].apply(
                lambda x: min(x['lower_depth'], target_max), axis=1)
            df_subset['overlap_length'] = df_subset['overlap_end'] - df_subset['overlap_start']
            df_subset['sample_length'] = df_subset['lower_depth'] - df_subset['upper_depth']
            df_subset['overlap_ratio'] = df_subset['overlap_length'] / df_subset['sample_length']
            
            # Sort by overlap ratio (descending) and remove unnecessary columns
            df_subset = df_subset.sort_values('overlap_ratio', ascending=False)
            df_subset = df_subset.drop(['overlap_start', 'overlap_end', 'overlap_length', 
                                     'sample_length', 'overlap_ratio'], axis=1)
            
            print(f"Found {len(df_subset)} records for {category}")
            categorized_data[category] = df_subset
        else:
            print(f"No records found for {category}")
            categorized_data[category] = pd.DataFrame()
    
    return categorized_data

def backup_existing_files():
    """Backup existing CSV files before replacement"""
    data_dir = Path('/root/WOSIS_enhancement/data')
    backup_dir = data_dir / 'backup_original'
    backup_dir.mkdir(exist_ok=True)
    
    folders = ['WOSIS_0_15cm', 'WOSIS_15_30cm', 'WOSIS_30_60cm', 'WOSIS_60_100cm']
    
    for folder in folders:
        folder_path = data_dir / folder
        if folder_path.exists():
            csv_file = folder_path / f"{folder}.csv"
            test_csv_file = folder_path / f"{folder}_test.csv"
            
            if csv_file.exists():
                backup_file = backup_dir / f"{folder}.csv"
                shutil.copy2(csv_file, backup_file)
                print(f"Backed up {csv_file} to {backup_file}")
            
            if test_csv_file.exists():
                backup_test_file = backup_dir / f"{folder}_test.csv"
                shutil.copy2(test_csv_file, backup_test_file)
                print(f"Backed up {test_csv_file} to {backup_test_file}")

def save_categorized_data(categorized_data):
    """Save categorized data to respective folders"""
    data_dir = Path('/root/lucica/WOSIS_enhancement/data')
    
    for category, df in categorized_data.items():
        folder_path = data_dir / category
        
        if len(df) > 0:
            # Save main CSV file
            csv_file = folder_path / f"{category}.csv"
            df.to_csv(csv_file, index=False)
            print(f"Saved {len(df)} records to {csv_file}")
            
            # Create a test subset (10% of data or minimum 100 records)
            test_size = max(100, len(df) // 10)
            df_test = df.sample(n=min(test_size, len(df)), random_state=42)
            
            test_csv_file = folder_path / f"{category}_test.csv"
            df_test.to_csv(test_csv_file, index=False)
            print(f"Saved {len(df_test)} test records to {test_csv_file}")
        else:
            print(f"No data to save for {category}")

def main():
    print("Starting WOSIS data splitting process...")
    
    # Load the large WOSIS dataset
    wosis_large_path = '/root/lucica/WOSIS_enhancement/data/wosis_large/wosis_large.csv'
    print(f"Loading data from {wosis_large_path}")
    
    df = pd.read_csv(wosis_large_path)
    print(f"Loaded {len(df)} total records")
    
    # Show date distribution
    print("\nDate distribution:")
    date_counts = df['date'].value_counts()
    print(f"Records with actual dates: {len(df[df['date'] != 'Unknown'])}")
    print(f"Records with 'Unknown' dates: {len(df[df['date'] == 'Unknown'])}")
    
    # Backup existing files
    print("\nBacking up existing files...")
    backup_existing_files()
    
    # Categorize data by depth
    print("\nCategorizing data by depth ranges...")
    categorized_data = categorize_by_depth(df)
    
    # Save categorized data
    print("\nSaving categorized data...")
    save_categorized_data(categorized_data)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_saved = sum(len(df) for df in categorized_data.values())
    print(f"Original data: {len(df)} records")
    print(f"Total categorized: {total_saved} records")
    
    for category, df_cat in categorized_data.items():
        if len(df_cat) > 0:
            date_info = df_cat['date'].value_counts()
            non_unknown = len(df_cat[df_cat['date'] != 'Unknown'])
            print(f"{category}: {len(df_cat)} records ({non_unknown} with dates)")
    
    print("\nData splitting completed successfully!")
    print("Original files backed up to: /root/lucica/WOSIS_enhancement/data/backup_original/")

if __name__ == "__main__":
    main() 