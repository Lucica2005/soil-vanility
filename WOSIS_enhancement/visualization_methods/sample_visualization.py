#!/usr/bin/env python3
"""
WOSIS Original Dataset Visualization

This script creates simplified visualizations of the original WOSIS soil data:
1. EC values distribution on world map
2. EC values histogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import sys
import glob
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('lucica/WOSIS_enhancement')

try:
    from lat_lon_to_ec_calculator import LatLonToECConverter
    CONVERTER_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: lat_lon_to_ec_calculator not found. Using basic coordinate conversion.")
    CONVERTER_AVAILABLE = False

def load_original_dataset(dataset_path):
    """
    Load the original WOSIS dataset
    
    Args:
        dataset_path: Path to the original WOSIS CSV file
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"\n📂 LOADING ORIGINAL WOSIS DATASET...")
    print("-" * 50)
    print(f"Dataset path: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Successfully loaded {len(df)} records")
        
        # Display basic info about the dataset
        print(f"\n📊 Dataset Information:")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Shape: {df.shape}")
        
        # Check for required columns
        required_cols = ['X', 'Y', 'value_avg']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Display data range
        print(f"\n🌍 Geographic Coverage:")
        print(f"   Longitude range: {df['X'].min():.2f} to {df['X'].max():.2f}")
        print(f"   Latitude range: {df['Y'].min():.2f} to {df['Y'].max():.2f}")
        
        print(f"\n⚡ EC Value Statistics:")
        print(f"   EC range: {df['value_avg'].min():.3f} to {df['value_avg'].max():.3f} dS/m")
        print(f"   EC mean±std: {df['value_avg'].mean():.3f}±{df['value_avg'].std():.3f} dS/m")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def create_simplified_visualization(df, output_path, title="WOSIS Original Dataset", save_modified=False):
    """
    Create simplified world map visualization with only EC distribution and histogram
    
    Args:
        df: DataFrame with WOSIS data
        output_path: Path to save visualization
        title: Title for the visualization
        save_modified: If True, save modified samples to CSV
    """
    print(f"\n🎨 CREATING SIMPLIFIED VISUALIZATION...")
    print("-" * 50)
    
    # Filter valid coordinates and EC values
    df_clean = df.dropna(subset=['X', 'Y', 'value_avg'])
    df_clean = df_clean[(df_clean['X'] >= -180) & (df_clean['X'] <= 180)]
    df_clean = df_clean[(df_clean['Y'] >= -90) & (df_clean['Y'] <= 90)]
    df_clean = df_clean[df_clean['value_avg'] >= 0]  # Remove negative EC values
    
    print(f"✅ {len(df_clean)} valid samples after filtering")
    
    if len(df_clean) == 0:
        print("❌ No valid samples found")
        return None
    
    # Initialize coordinate converter if available
    converter = None
    if CONVERTER_AVAILABLE:
        try:
            converter = LatLonToECConverter()
            print("✅ Coordinate converter initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize converter: {e}")
            converter = None
    
    # Convert coordinates to pixel positions
    pixel_coords = []
    ec_values = []
    valid_rows = []
    modified_rows = []  # Store modified rows for saving
    
    print("🔄 Converting coordinates to pixels...")
    
    if converter:
        # Use the proper coordinate-to-pixel conversion
        for idx, row in df_clean.iterrows():
            try:
                lat, lon = row['Y'], row['X']  # Y=latitude, X=longitude
                x_pixel, y_pixel = converter.lat_lon_to_xy(lat, lon)
                ec = row['value_avg']
                true_ec = converter.calculate_ec_from_coordinates(lat, lon)
                
                # Check if pixel coordinates are within reasonable bounds
                if 0 <= x_pixel <= 2500 and 0 <= y_pixel <= 1600 and ec > 0 and true_ec > 0:
                    pixel_coords.append((x_pixel, y_pixel))
                    
                    # Calculate average EC value
                    if save_modified:
                        avg_ec = (ec + true_ec) / 2.0
                        ec_values.append(avg_ec)
                    else:
                        ec_values.append(ec)
                    
                    # Create modified row with averaged EC
                    modified_row = row.copy()
                    modified_row['value_avg'] = avg_ec
                    valid_rows.append(modified_row)
                    
                    # Store for saving if requested
                    if save_modified:
                        modified_rows.append(modified_row)
            except Exception as e:
                continue
    else:
        # Fallback: simple linear mapping (less accurate)
        print("⚠️ Using fallback coordinate mapping")
        for idx, row in df_clean.iterrows():
            try:
                lat, lon = row['Y'], row['X']
                # Simple linear mapping to approximate pixel coordinates
                x_pixel = int((lon + 180) * (2000 / 360))  # Map -180,180 to 0,2000
                y_pixel = int((90 - lat) * (1200 / 180))   # Map -90,90 to 1200,0
                
                if 0 <= x_pixel <= 2000 and 0 <= y_pixel <= 1200:
                    pixel_coords.append((x_pixel, y_pixel))
                    ec_values.append(row['value_avg'])
                    valid_rows.append(row)
            except Exception as e:
                continue
    
    print(f"✅ Successfully converted {len(pixel_coords)} coordinates to pixels")
    
    # Save modified samples if requested
    if save_modified and modified_rows:
        # Create DataFrame from modified rows
        modified_df = pd.DataFrame(modified_rows)
        
        # Determine output path for modified CSV
        csv_output_path = output_path.replace(".png", "_modified.csv")
        
        # Save to CSV
        modified_df.to_csv(csv_output_path, index=False)
        print(f"💾 Saved {len(modified_rows)} modified samples to: {csv_output_path}")
    
    if len(pixel_coords) == 0:
        print("❌ No valid pixel coordinates found")
        return None
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ... rest of the visualization code remains unchanged ...
    
    # 1. World map with EC points overlaid on actual world map image
    world_map_path = 'lucica/WOSIS_enhancement/world_map.jpg'
    try:
        world_img = plt.imread(world_map_path)
        ax1.imshow(world_img, extent=[0, world_img.shape[1], world_img.shape[0], 0])
        print(f"✅ World map loaded as background: {world_img.shape}")
        
        # Adjust coordinate system to match image
        ax1.set_xlim(0, world_img.shape[1])
        ax1.set_ylim(world_img.shape[0], 0)  # Flip Y axis for image coordinates
        
    except Exception as e:
        print(f"⚠️ Could not load world map image: {e}")
        ax1.set_xlim(0, 2312)
        ax1.set_ylim(1536, 0)
    
    # Plot EC points with color intensity based on EC value
    x_coords = [coord[0] for coord in pixel_coords]
    y_coords = [coord[1] for coord in pixel_coords]
    
    # Create scatter plot with colors based on EC values
    scatter = ax1.scatter(x_coords, y_coords, c=ec_values, s=15, 
                         cmap='Reds', alpha=0.8, edgecolors='black', linewidth=0.1)
    
    ax1.set_title(f'{title} - Global EC Distribution\n'
                 f'({len(pixel_coords):,} soil samples)', 
                 fontsize=16, fontweight='bold')
    ax1.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax1.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax1.axis('off')  # Hide axes for cleaner look
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8, aspect=30)
    cbar1.set_label('EC Value (dS/m)', fontsize=12)
    
    # 2. EC value histogram
    ax2.hist(ec_values, bins=50, alpha=0.8, color='darkred', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('EC Value (dS/m)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('EC Value Distribution', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    # Add statistics text box
    stats_text = (f'Samples: {len(ec_values):,}\n'
                 f'Mean: {np.mean(ec_values):.3f} dS/m\n'
                 f'Median: {np.median(ec_values):.3f} dS/m\n'
                 f'Std: {np.std(ec_values):.3f} dS/m\n'
                 f'Min: {np.min(ec_values):.3f} dS/m\n'
                 f'Max: {np.max(ec_values):.3f} dS/m')
    ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9),
             verticalalignment='top', fontsize=11)
    
    plt.tight_layout(pad=3.0)
    
    # Save visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")
    
    # Return summary statistics
    return {
        'total_samples': len(df),
        'valid_samples': len(df_clean),
        'pixel_converted': len(pixel_coords),
        'ec_stats': {
            'mean': np.mean(ec_values),
            'median': np.median(ec_values),
            'std': np.std(ec_values),
            'min': np.min(ec_values),
            'max': np.max(ec_values)
        }
    }

def main():
    """
    Main function to run the simplified visualization
    """
    print("🚀 STARTING WOSIS SIMPLIFIED VISUALIZATION")
    print("=" * 60)
    
    
    name="WOSIS_0_15cm"
    epoch=8000

    save_modified=False
    
    dataset_path = f"/root/WOSIS_enhancement/tabdiff/result/{name}/{name}_ec/{epoch}/samples.csv"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the dataset file exists at the specified location.")
        return
    
    # 1. Load the original dataset
    df = load_original_dataset(dataset_path)
    
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # 2. Create simplified visualization
    print("\n🎨 CREATING SIMPLIFIED VISUALIZATION...")
    main_output = f"visualizations/{name}_samples_{epoch}.png"
    result = create_simplified_visualization(df, main_output, f"{name} Generated Samples",save_modified)
    
    # 3. Print summary
    if result:
        print("\n📋 VISUALIZATION SUMMARY")
        print("=" * 30)
        print(f"📊 Dataset: WOSIS 0-15cm depth")
        print(f"📈 Total records: {result['total_samples']:,}")
        print(f"✅ Valid samples: {result['valid_samples']:,}")
        print(f"🎯 Pixel converted: {result['pixel_converted']:,}")
        print(f"⚡ EC statistics:")
        print(f"   Range: {result['ec_stats']['min']:.3f} - {result['ec_stats']['max']:.3f} dS/m")
        print(f"   Mean: {result['ec_stats']['mean']:.3f} dS/m")
        print(f"   Median: {result['ec_stats']['median']:.3f} dS/m")
        print(f"   Std Dev: {result['ec_stats']['std']:.3f} dS/m")
    
    print("\n✅ Simplified visualization completed!")
    print("📁 Check the 'visualizations/' directory for output file:")
    print(f"   • {main_output}")
    
    print("\n🎯 VISUALIZATION INCLUDES:")
    print("-" * 25)
    print("✅ Global EC distribution on world map")
    print("✅ EC value histogram with statistics")
    print("✅ Clean, focused presentation")

if __name__ == "__main__":
    main() 