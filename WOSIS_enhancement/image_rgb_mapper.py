#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World Map RGB Mapper
Loads world_map.jpg and saves RGB mapping of each pixel as numpy array
"""

import numpy as np
from PIL import Image
import os

def save_world_map_rgb_mapping(image_path='world_map.jpg', output_path='world_map_rgb.npy'):
    """
    Load world map image and save RGB mapping as numpy array
    
    Args:
        image_path (str): Path to world map image
        output_path (str): Path to save numpy array
        
    Returns:
        numpy.ndarray: RGB array with shape (height, width, 3)
    """
    try:
        # Load image
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        rgb_array = np.array(image)
        
        print(f"Image shape: {rgb_array.shape}")
        print(f"Image size: {rgb_array.shape[1]} x {rgb_array.shape[0]} pixels")
        
        # Save as numpy array
        np.save(output_path, rgb_array)
        print(f"RGB mapping saved to: {output_path}")
        
        return rgb_array
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found!")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_world_map_rgb_mapping(array_path='world_map_rgb.npy'):
    """
    Load previously saved RGB mapping
    
    Args:
        array_path (str): Path to numpy array file
        
    Returns:
        numpy.ndarray: RGB array with shape (height, width, 3)
    """
    try:
        rgb_array = np.load(array_path)
        print(f"RGB mapping loaded from: {array_path}")
        print(f"Array shape: {rgb_array.shape}")
        return rgb_array
    except FileNotFoundError:
        print(f"Error: RGB mapping file '{array_path}' not found!")
        print("Please run save_world_map_rgb_mapping() first.")
        return None
    except Exception as e:
        print(f"Error loading RGB mapping: {e}")
        return None

def get_rgb_at_position(rgb_array, x, y):
    """
    Get RGB values at specific pixel position
    
    Args:
        rgb_array (numpy.ndarray): RGB array from world map
        x (int): X coordinate (column)
        y (int): Y coordinate (row)
        
    Returns:
        tuple: (R, G, B) values or None if out of bounds
    """
    if rgb_array is None:
        return None
        
    height, width, _ = rgb_array.shape
    
    # Check bounds
    if x < 0 or x >= width or y < 0 or y >= height:
        print(f"Warning: Position ({x}, {y}) is out of bounds (image size: {width} x {height})")
        return None
    
    # Get RGB values
    r, g, b = rgb_array[y, x]  # Note: numpy uses [row, col] indexing
    return int(r), int(g), int(b)

if __name__ == "__main__":
    # Save RGB mapping from world map image
    rgb_array = save_world_map_rgb_mapping()
    
    if rgb_array is not None:
        print("\nTesting RGB extraction at sample positions:")
        test_positions = [(100, 100), (500, 300), (1000, 600)]
        
        for x, y in test_positions:
            rgb = get_rgb_at_position(rgb_array, x, y)
            if rgb:
                r, g, b = rgb
                print(f"Position ({x}, {y}): RGB({r}, {g}, {b})") 