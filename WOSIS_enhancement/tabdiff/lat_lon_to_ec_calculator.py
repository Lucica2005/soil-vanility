#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latitude/Longitude to EC Calculator
Converts geographic coordinates to EC values using world map RGB data
"""

import numpy as np
import sys
import os

# Import required modules
from image_rgb_mapper import load_world_map_rgb_mapping, get_rgb_at_position
from rgb_to_ec_calculator import rgb_to_ec

class LatLonToECConverter:
    """
    Converter class for transforming latitude/longitude coordinates to EC values
    """
    
    def __init__(self, rgb_mapping_path='world_map_rgb.npy'):
        """
        Initialize the converter
        
        Args:
            rgb_mapping_path (str): Path to the saved RGB mapping numpy array
        """
        # Reference points for coordinate transformation
        # Point 1: (x, y, lat, lon) = (1150, 819, 36.6232, 11.1788)
        # Point 2: (x, y, lat, lon) = (857, 1109, -3.3291, -39.1475)
        
        self.ref_point1 = {'x': 1150, 'y': 819, 'lat': 36.6232, 'lon': 11.1788}
        self.ref_point2 = {'x': 857, 'y': 1109, 'lat': -3.3291, 'lon': -39.1475}
        
        # Calculate transformation coefficients
        self._calculate_transformation_coefficients()
        
        # Load RGB mapping
        self.rgb_array = load_world_map_rgb_mapping(rgb_mapping_path)
        if self.rgb_array is None:
            print("Warning: RGB mapping not loaded. Please run image_rgb_mapper.py first.")
    
    def _calculate_transformation_coefficients(self):
        """
        Calculate linear transformation coefficients from lat/lon to x/y
        
        Transformations:
        x = m_x * lon + b_x
        y = m_y * lat + b_y
        """
        # Calculate longitude to x transformation coefficients
        lon1, lon2 = self.ref_point1['lon'], self.ref_point2['lon']
        x1, x2 = self.ref_point1['x'], self.ref_point2['x']
        
        self.m_x = (x1 - x2) / (lon1 - lon2)
        self.b_x = x1 - self.m_x * lon1
        
        # Calculate latitude to y transformation coefficients  
        lat1, lat2 = self.ref_point1['lat'], self.ref_point2['lat']
        y1, y2 = self.ref_point1['y'], self.ref_point2['y']
        
        self.m_y = (y1 - y2) / (lat1 - lat2)
        self.b_y = y1 - self.m_y * lat1
        
        print(f"Transformation coefficients calculated:")
        print(f"  x = {self.m_x:.6f} * lon + {self.b_x:.6f}")
        print(f"  y = {self.m_y:.6f} * lat + {self.b_y:.6f}")
        
        # Verify with reference points
        self._verify_transformation()
    
    def _verify_transformation(self):
        """
        Verify transformation accuracy with reference points
        """
        print(f"\nVerification:")
        
        for i, ref_point in enumerate([self.ref_point1, self.ref_point2], 1):
            calc_x = self.m_x * ref_point['lon'] + self.b_x
            calc_y = self.m_y * ref_point['lat'] + self.b_y
            
            print(f"  Point {i}: Expected ({ref_point['x']}, {ref_point['y']}), "
                  f"Calculated ({calc_x:.1f}, {calc_y:.1f})")
    
    def lat_lon_to_xy(self, latitude, longitude):
        """
        Convert latitude/longitude to x/y pixel coordinates
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            tuple: (x, y) pixel coordinates as integers (rounded to nearest pixel)
        """
        x = self.m_x * longitude + self.b_x
        y = self.m_y * latitude + self.b_y
        
        # Round to nearest pixel
        x_pixel = int(round(x))
        y_pixel = int(round(y))
        
        return x_pixel, y_pixel
    
    def get_rgb_from_coordinates(self, latitude, longitude):
        """
        Get RGB values from latitude/longitude coordinates
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            tuple: (R, G, B) values or None if out of bounds
        """
        if self.rgb_array is None:
            print("Error: RGB mapping not Navailable")
            return None
        
        # Convert to pixel coordinates
        x, y = self.lat_lon_to_xy(latitude, longitude)
        
        # Get RGB values
        rgb = get_rgb_at_position(self.rgb_array, x, y)
        
        return rgb
    
    def calculate_ec_from_coordinates(self, latitude, longitude, verbose=True):
        """
        Calculate EC value from latitude/longitude coordinates
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            verbose (bool): Whether to print detailed information
            
        Returns:
            float: EC value or None if calculation fails
        """
        if verbose:
            print(f"\nCalculating EC for coordinates ({latitude}, {longitude}):")
        
        # Step 1: Convert to pixel coordinates
        x, y = self.lat_lon_to_xy(latitude, longitude)
        if verbose:
            print(f"  Pixel coordinates: ({x}, {y})")
        
        # Step 2: Get RGB values
        rgb = self.get_rgb_from_coordinates(latitude, longitude)
        if rgb is None:
            if verbose:
                print("  Error: Could not retrieve RGB values")
            return None
        
        r, g, b = rgb
        if verbose:
            print(f"  RGB values: ({r}, {g}, {b})")
        
        # Step 3: Calculate EC value
        ec_value = rgb_to_ec(r, g, b)
        if verbose:
            print(f"  EC value: {ec_value}")
        
        return ec_value

def batch_calculate_ec(coordinates_list, converter=None):
    """
    Calculate EC values for a batch of coordinates
    
    Args:
        coordinates_list (list): List of (latitude, longitude) tuples
        converter (LatLonToECConverter): Converter instance (creates new if None)
        
    Returns:
        list: List of (latitude, longitude, ec_value) tuples
    """
    if converter is None:
        converter = LatLonToECConverter()
    
    results = []
    
    print(f"\nBatch processing {len(coordinates_list)} coordinates:")
    print("-" * 60)
    
    for i, (lat, lon) in enumerate(coordinates_list, 1):
        print(f"Processing {i}/{len(coordinates_list)}: ({lat}, {lon})")
        
        ec_value = converter.calculate_ec_from_coordinates(lat, lon, verbose=True)
        results.append((lat, lon, ec_value))
        
        if ec_value is not None:
            print(f"  -> EC = {ec_value}")
        else:
            print(f"  -> EC calculation failed")
    
    return results

def interactive_ec_calculator():
    """
    Interactive tool for calculating EC values from coordinates
    """
    print("=== Interactive Latitude/Longitude to EC Calculator ===")
    print("Enter latitude and longitude coordinates to calculate EC values")
    print("Type 'quit' to exit")
    
    # Initialize converter
    converter = LatLonToECConverter()
    
    if converter.rgb_array is None:
        print("Error: Could not load RGB mapping. Please run image_rgb_mapper.py first.")
        return
    
    while True:
        try:
            user_input = input("\nEnter coordinates (format: latitude longitude): ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            
            # Parse input
            coords = user_input.split()
            if len(coords) != 2:
                print("Please enter exactly 2 numbers (latitude longitude)")
                continue
            
            latitude, longitude = map(float, coords)
            
            # Calculate EC
            ec_value = converter.calculate_ec_from_coordinates(latitude, longitude)
            
            if ec_value is not None:
                print(f"\nResult: Coordinates ({latitude}, {longitude}) -> EC = {ec_value}")
            else:
                print(f"\nError: Could not calculate EC for coordinates ({latitude}, {longitude})")
                
        except ValueError:
            print("Please enter valid numbers")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def main():
    """
    Main function with example usage
    """
    print("=== Latitude/Longitude to EC Calculator ===")
    
    # Initialize converter
    converter = LatLonToECConverter()
    
    if converter.rgb_array is None:
        print("Error: RGB mapping not available. Please run image_rgb_mapper.py first.")
        return
    
    # Test with reference points to verify system works
    print("\n=== Testing with reference points ===")
    test_coordinates = [
        (36.6232, 11.1788),   # Reference point 1
        (-3.3291, -39.1475),  # Reference point 2
        (0.0, 0.0),           # Equator, Prime Meridian
        (40.7128, -74.0060),  # New York City
        (51.5074, -0.1278),   # London
        (-33.8688, 151.2093), # Sydney
    ]
    
    results = batch_calculate_ec(test_coordinates, converter)
    
    print("\n=== Summary of Results ===")
    for lat, lon, ec in results:
        print(f"({lat:8.4f}, {lon:9.4f}) -> EC = {ec}")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Optionally run interactive calculator
    print("\n" + "="*60)
    run_interactive = input("Run interactive calculator? (y/n): ").strip().lower()
    if run_interactive in ['y', 'yes']:
        interactive_ec_calculator() 