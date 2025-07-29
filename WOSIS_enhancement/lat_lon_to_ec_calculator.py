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
    
    def __init__(self, rgb_mapping_path='/root/lucica/WOSIS_enhancement/world_map_rgb.npy'):
        """
        Initialize the converter
        
        Args:
            rgb_mapping_path (str): Path to the saved RGB mapping numpy array
        """
        # Reference points for coordinate transformation (5 points for optimization)
        # Point 1: (x, y, lat, lon) = (1150, 819, 36.6232, 11.1788)
        # Point 2: (x, y, lat, lon) = (857, 1109, -3.3291, -39.1475)
        # Point 3: (x, y, lat, lon) = (1996, 1145, -11.5467, 142.4657)
        # Point 4: (x, y, lat, lon) = (1827, 1212, -21.3581, 115.8594)
        # Point 5: (x, y, lat, lon) = (380, 921, 22.5135, -109.3151)
        
        self.ref_points = [
            {'x': 1150, 'y': 819, 'lat': 36.6232, 'lon': 11.1788},
            {'x': 857, 'y': 1109, 'lat': -3.3291, 'lon': -39.1475},
            {'x': 1996, 'y': 1145, 'lat': -11.5467, 'lon': 142.4657},
            {'x': 1827, 'y': 1212, 'lat': -21.3581, 'lon': 115.8594},
            {'x': 380, 'y': 921, 'lat': 22.5135, 'lon': -109.3151}
        ]
        
        # Calculate transformation coefficients
        self._calculate_transformation_coefficients()
        
        # Load RGB mapping
        self.rgb_array = load_world_map_rgb_mapping(rgb_mapping_path)
        if self.rgb_array is None:
            print("Warning: RGB mapping not loaded. Please run image_rgb_mapper.py first.")
    
    def _calculate_transformation_coefficients(self):
        """
        Calculate transformation coefficients using 5 reference points with Mercator projection
        
        Uses linear transformation for longitude and Mercator + quadratic for latitude
        with least squares fitting for optimal parameters
        """
        import math
        import numpy as np
        
        # Extract coordinates from reference points
        lats = np.array([p['lat'] for p in self.ref_points])
        lons = np.array([p['lon'] for p in self.ref_points])
        xs = np.array([p['x'] for p in self.ref_points])
        ys = np.array([p['y'] for p in self.ref_points])
        
        # For longitude: linear fit using least squares (x = m_x * lon + b_x)
        A_lon = np.column_stack([lons, np.ones(len(lons))])
        lon_coeffs = np.linalg.lstsq(A_lon, xs, rcond=None)[0]
        self.m_x, self.b_x = lon_coeffs[0], lon_coeffs[1]
        
        # For latitude: Mercator projection + quadratic fit using least squares
        # First convert latitudes to Mercator y coordinates
        def lat_to_mercator_y(lat_deg):
            lat_rad = math.radians(lat_deg)
            lat_rad = max(-1.4, min(1.4, lat_rad))  # Avoid poles
            return math.log(math.tan(math.pi / 4 + lat_rad / 2))
        
        merc_ys = np.array([lat_to_mercator_y(lat) for lat in lats])
        
        # Quadratic fit: y = a * merc_y^2 + b * merc_y + c
        A_lat = np.column_stack([merc_ys**2, merc_ys, np.ones(len(merc_ys))])
        lat_coeffs = np.linalg.lstsq(A_lat, ys, rcond=None)[0]
        self.a_y, self.b_y, self.c_y = lat_coeffs[0], lat_coeffs[1], lat_coeffs[2]
        
        print(f"Optimized transformation coefficients calculated (5 points):")
        print(f"  Longitude (linear): x = {self.m_x:.6f} * lon + {self.b_x:.6f}")
        print(f"  Latitude (Mercator + quadratic): y = {self.a_y:.6f} * merc_y^2 + {self.b_y:.6f} * merc_y + {self.c_y:.6f}")
        
        # Verify with all reference points
        self._verify_transformation()
    
    def _verify_transformation(self):
        """
        Verify transformation accuracy with all 5 reference points
        """
        print(f"\nTransformation verification (5 points):")
        
        import math
        
        def lat_to_mercator_y(lat_deg):
            lat_rad = math.radians(lat_deg)
            lat_rad = max(-1.4, min(1.4, lat_rad))
            return math.log(math.tan(math.pi / 4 + lat_rad / 2))
        
        for i, ref_point in enumerate(self.ref_points, 1):
            # Calculate using our transformation
            lon = ref_point['lon']
            calc_x = self.m_x * lon + self.b_x
            
            # Mercator transformation for latitude
            lat = ref_point['lat']
            y_merc = lat_to_mercator_y(lat)
            calc_y = self.a_y * y_merc**2 + self.b_y * y_merc + self.c_y
            
            error_x = abs(calc_x - ref_point['x'])
            error_y = abs(calc_y - ref_point['y'])
            total_error = (error_x**2 + error_y**2)**0.5
            
            print(f"  Point {i}: Expected ({ref_point['x']}, {ref_point['y']}), "
                  f"Calculated ({calc_x:.1f}, {calc_y:.1f}), "
                  f"Error ({error_x:.1f}, {error_y:.1f}), Total: {total_error:.1f}")
    
    def lat_lon_to_xy(self, latitude, longitude):
        """
        Convert latitude/longitude to x/y pixel coordinates using optimized Mercator method
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            tuple: (x, y) pixel coordinates as integers (rounded to nearest pixel)
        """
        import math
        
        # Longitude transformation (linear)
        x = self.m_x * longitude + self.b_x
        
        # Latitude transformation (Mercator + quadratic)
        def lat_to_mercator_y(lat_deg):
            lat_rad = math.radians(lat_deg)
            lat_rad = max(-1.4, min(1.4, lat_rad))
            return math.log(math.tan(math.pi / 4 + lat_rad / 2))
        
        y_merc = lat_to_mercator_y(latitude)
        y = self.a_y * y_merc**2 + self.b_y * y_merc + self.c_y
        
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
            
            #    print("Error: RGB mapping not Navailable")
            return None
        
        # Convert to pixel coordinates
        x, y = self.lat_lon_to_xy(latitude, longitude)
        
        # Get RGB values
        rgb = get_rgb_at_position(self.rgb_array, x, y)
        
        return rgb
    
    def calculate_ec_from_coordinates(self, latitude, longitude, verbose=False):
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