#!/usr/bin/env python3
"""
Differentiable EC Calculator for PyTorch tensors
This module provides differentiable versions of EC calculation functions
"""

import torch
import torch.nn.functional as F
import numpy as np

class DifferentiableECCalculator:
    """
    Differentiable EC calculator that preserves gradient computationfi
    """
    
    def __init__(self, rgb_array_path='/root/lucica/WOSIS_enhancement/world_map_rgb.npy', device='cuda'):
        self.device = device
        self.rgb_tensor = None
        
        # Reference points for coordinate transformation (5 points for optimization)
        # Point 1: (x, y, lat, lon) = (1150, 819, 36.6232, 11.1788)
        # Point 2: (x, y, lat, lon) = (857, 1109, -3.3291, -39.1475)
        # Point 3: (x, y, lat, lon) = (1996, 1145, -11.5467, 142.4657)
        # Point 4: (x, y, lat, lon) = (1827, 1212, -21.3581, 115.8594)
        # Point 5: (x, y, lat, lon) = (380, 921, 22.5135, -109.3151)
        self.ref_points = torch.tensor([
            [1150.0, 819.0, 36.6232, 11.1788],    # [x1, y1, lat1, lon1]
            [857.0, 1109.0, -3.3291, -39.1475],   # [x2, y2, lat2, lon2]
            [1996.0, 1145.0, -11.5467, 142.4657], # [x3, y3, lat3, lon3]
            [1827.0, 1212.0, -21.3581, 115.8594], # [x4, y4, lat4, lon4]
            [380.0, 921.0, 22.5135, -109.3151]    # [x5, y5, lat5, lon5]
        ], device=device, dtype=torch.float32)
        
        # Calculate transformation coefficients
        self._calculate_transformation_coefficients()
        
        # Load RGB array if provided
        if rgb_array_path:
            self.load_rgb_array(rgb_array_path)
    
    def load_rgb_array(self, rgb_array_path):
        """Load RGB array and convert to PyTorch tensor"""
        try:
            if rgb_array_path.endswith('.npy'):
                rgb_array = np.load(rgb_array_path)
            else:
                from PIL import Image
                img = Image.open(rgb_array_path)
                rgb_array = np.array(img)
            
            # Convert to PyTorch tensor and move to device
            # Set requires_grad=True to enable gradient computation through the RGB tensor
            self.rgb_tensor = torch.from_numpy(rgb_array).float().to(self.device).requires_grad_(True)
            #print(f"Loaded RGB tensor with shape: {self.rgb_tensor.shape}")
            
        except Exception as e:
            print(f"Warning: Could not load RGB array from {rgb_array_path}: {e}")
            # Create a dummy RGB tensor for testing with requires_grad=True
            self.rgb_tensor = (torch.randn(1536, 2312, 3, device=self.device) * 127 + 127).requires_grad_(True)
    
    def _calculate_transformation_coefficients(self):
        """
        Calculate transformation coefficients using 5 reference points with Mercator projection
        
        Uses linear transformation for longitude and Mercator + quadratic for latitude
        with least squares fitting for optimal parameters
        """
        # Extract coordinates from reference points
        xs = self.ref_points[:, 0]  # x coordinates
        ys = self.ref_points[:, 1]  # y coordinates  
        lats = self.ref_points[:, 2]  # latitudes
        lons = self.ref_points[:, 3]  # longitudes
        
        # For longitude: linear fit using least squares (x = m_x * lon + b_x)
        A_lon = torch.stack([lons, torch.ones_like(lons)], dim=1)
        lon_coeffs = torch.linalg.lstsq(A_lon, xs).solution
        self.m_x, self.b_x = lon_coeffs[0], lon_coeffs[1]
        
        # For latitude: Mercator projection + quadratic fit using least squares
        # First convert latitudes to Mercator y coordinates - same as lat_lon_to_ec_calculator.py
        lats_rad = torch.deg2rad(lats)
        # Use max/min like in lat_lon_to_ec_calculator.py instead of torch.clamp
        lats_clamped = torch.where(
            lats_rad > 1.4, 
            torch.full_like(lats_rad, 1.4),
            torch.where(lats_rad < -1.4, torch.full_like(lats_rad, -1.4), lats_rad)
        )
        merc_ys = torch.log(torch.tan(torch.pi / 4 + lats_clamped / 2))
        
        # Quadratic fit: y = a * merc_y^2 + b * merc_y + c
        A_lat = torch.stack([merc_ys**2, merc_ys, torch.ones_like(merc_ys)], dim=1)
        lat_coeffs = torch.linalg.lstsq(A_lat, ys).solution
        self.a_y, self.b_y, self.c_y = lat_coeffs[0], lat_coeffs[1], lat_coeffs[2]
        
        print(f"Optimized transformation coefficients calculated (5 points):")
        print(f"  Longitude (linear): x = {self.m_x:.6f} * lon + {self.b_x:.6f}")
        print(f"  Latitude (Mercator + quadratic): y = {self.a_y:.6f} * merc_y^2 + {self.b_y:.6f} * merc_y + {self.c_y:.6f}")
        
        # Verify transformation
        self._verify_transformation()
    
    def _verify_transformation(self):
        """
        Verify transformation accuracy with all 5 reference points
        """
        print(f"\nTransformation verification (5 points):")
        
        for i in range(self.ref_points.shape[0]):
            x_expected, y_expected, lat, lon = self.ref_points[i]
            
            # Calculate using our transformation
            x_calc = self.m_x * lon + self.b_x
            
            # Mercator transformation for latitude - same as lat_lon_to_ec_calculator.py
            lat_rad = torch.deg2rad(lat)
            # Use max/min like in lat_lon_to_ec_calculator.py instead of torch.clamp
            lat_clamped = torch.where(
                lat_rad > 1.4, 
                torch.full_like(lat_rad, 1.4),
                torch.where(lat_rad < -1.4, torch.full_like(lat_rad, -1.4), lat_rad)
            )
            merc_y = torch.log(torch.tan(torch.pi / 4 + lat_clamped / 2))
            y_calc = self.a_y * merc_y**2 + self.b_y * merc_y + self.c_y
            
            error_x = torch.abs(x_calc - x_expected)
            error_y = torch.abs(y_calc - y_expected)
            total_error = torch.sqrt(error_x**2 + error_y**2)
            
            print(f"  Point {i+1}: Expected ({x_expected:.0f}, {y_expected:.0f}), "
                  f"Calculated ({x_calc:.1f}, {y_calc:.1f}), "
                  f"Error ({error_x:.1f}, {error_y:.1f}), Total: {total_error:.1f}")

    def lat_lon_to_xy_tensor(self, lat_coords, lon_coords):
        """
        Convert lat/lon coordinates to pixel coordinates using optimized Mercator method
        
        Uses linear transformation for longitude and Mercator + quadratic for latitude
        Optimized using 5 reference points for better accuracy
        This method follows the same logic as lat_lon_to_ec_calculator.py
        
        Args:
            lat_coords: Tensor of latitude coordinates [batch_size]
            lon_coords: Tensor of longitude coordinates [batch_size]
            
        Returns:
            x_coords, y_coords: Pixel coordinates as tensors
        """
        # Longitude transformation (linear) - same as lat_lon_to_ec_calculator.py
        x_coords = self.m_x * lon_coords + self.b_x
        
        # Latitude transformation (Mercator + quadratic) - same as lat_lon_to_ec_calculator.py
        lat_coords_rad = torch.deg2rad(lat_coords)
        # Use max/min like in lat_lon_to_ec_calculator.py instead of torch.clamp
        lat_coords_clamped = torch.where(
            lat_coords_rad > 1.4, 
            torch.full_like(lat_coords_rad, 1.4),
            torch.where(lat_coords_rad < -1.4, torch.full_like(lat_coords_rad, -1.4), lat_coords_rad)
        )
        merc_ys = torch.log(torch.tan(torch.pi / 4 + lat_coords_clamped / 2))
        
        # Apply quadratic transformation
        y_coords = self.a_y * merc_ys**2 + self.b_y * merc_ys + self.c_y
        
        return x_coords, y_coords
    
    def get_rgb_from_coordinates_tensor(self, lat_coords, lon_coords):
        """
        Get RGB values from coordinates using differentiable operations
        
        Args:
            lat_coords: Tensor of latitude coordinates [batch_size]
            lon_coords: Tensor of longitude coordinates [batch_size]
            
        Returns:
            rgb_values: Tensor of RGB values [batch_size, 3]
        """
        if self.rgb_tensor is None:
            # Return dummy values if no RGB tensor loaded - ensure they connect to input gradients
            batch_size = lat_coords.shape[0]
            # Make dummy values dependent on input coordinates to preserve gradients
            dummy_rgb = torch.stack([
                lat_coords * 0.0 + 127.0,  # R channel
                lon_coords * 0.0 + 127.0,  # G channel  
                (lat_coords + lon_coords) * 0.0 + 127.0  # B channel
            ], dim=1)
            return dummy_rgb
        
        # Convert to pixel coordinates
        x_coords, y_coords = self.lat_lon_to_xy_tensor(lat_coords, lon_coords)
        #print("x,y pixels:",x_coords,y_coords)
        # Clamp coordinates to valid range
        height, width = self.rgb_tensor.shape[:2]
        x_coords = torch.clamp(x_coords, 0, width - 1)
        y_coords = torch.clamp(y_coords, 0, height - 1)
        
        # Use bilinear interpolation for differentiable sampling
        # This preserves gradients better than direct indexing
        x_norm = 2.0 * x_coords / (width - 1) - 1.0
        y_norm = 2.0 * y_coords / (height - 1) - 1.0
        
        # Create grid for sampling [batch_size, 1, 1, 2]
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1).unsqueeze(1)
        
        # Prepare RGB tensor for sampling [1, 3, height, width]
        rgb_for_sampling = self.rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Sample RGB values using bilinear interpolation
        batch_size = lat_coords.shape[0]
        grid_expanded = grid.expand(batch_size, -1, -1, -1)
        
        sampled_rgb = F.grid_sample(
            rgb_for_sampling.expand(batch_size, -1, -1, -1),
            grid_expanded,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Extract RGB values correctly to get [batch_size, 3]
        # sampled_rgb shape: [batch_size, 3, 1, 1]
        rgb_values = sampled_rgb.squeeze(-1).squeeze(-1)  # Now [batch_size, 3]
        
        return rgb_values
    def get_rgb_from_pixels(self, x_coords, y_coords):
        height, width = self.rgb_tensor.shape[:2]
        x_coords = torch.clamp(x_coords, 0, width - 1)
        y_coords = torch.clamp(y_coords, 0, height - 1)
        
        # Use bilinear interpolation for differentiable sampling
        # This preserves gradients better than direct indexing
        x_norm = 2.0 * x_coords / (width - 1) - 1.0
        y_norm = 2.0 * y_coords / (height - 1) - 1.0
        
        # Create grid for sampling [batch_size, 1, 1, 2]
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1).unsqueeze(1)
        
        # Prepare RGB tensor for sampling [1, 3, height, width]
        rgb_for_sampling = self.rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Sample RGB values using bilinear interpolation
        
        grid_expanded = grid.expand(1, -1, -1, -1)
        
        sampled_rgb = F.grid_sample(
            rgb_for_sampling.expand(1, -1, -1, -1),
            grid_expanded,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # Extract RGB values correctly to get [batch_size, 3]
        # sampled_rgb shape: [batch_size, 3, 1, 1]
        rgb_values = sampled_rgb.squeeze(-1).squeeze(-1)  # Now [batch_size, 3]
        
        return rgb_values
    
    def rgb_to_ec_tensor(self, rgb_values):
        """
        Convert RGB values to EC using differentiable operations
        
        Args:
            rgb_values: Tensor of RGB values [batch_size, 3]
            
        Returns:
            ec_values: Tensor of EC values [batch_size]
        """
        r, g, b = rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2]
        
        # Initialize EC values
        ec_values = torch.zeros_like(r)
        
        # Rule 1: B >= 200 -> EC = -1
        mask_b_high = b >= 200
        ec_values = torch.where(mask_b_high, torch.full_like(ec_values, 0.0), ec_values)
        
        # Rule 2: B < 200
        mask_b_low = b < 200
        
        # Case 1: G >= 245, EC = r * 4.0 / 255.0
        mask_case1 = mask_b_low & (g >= 245)
        ec_case1 = r * 4.0 / 255.0
        ec_values = torch.where(mask_case1, ec_case1, ec_values)
        
        # Case 2: R >= 245, EC = 4.0 + (255 - g) * 8.0 / 255.0
        mask_case2 = mask_b_low & (r >= 245) & ~mask_case1
        ec_case2 = 4.0 + (255.0 - g) * 8.0 / 255.0
        ec_values = torch.where(mask_case2, ec_case2, ec_values)
        
        # Case 3: G <= 10, EC = 12.0 + (255 - r) * 18.0 / 255.0
        mask_case3 = mask_b_low & (g <= 10) & ~mask_case1 & ~mask_case2
        ec_case3 = 12.0 + (255.0 - r) * 18.0 / 255.0
        ec_values = torch.where(mask_case3, ec_case3, ec_values)
        
        # Case 4: R <= 10, EC = 30 + g/3
        mask_case4 = mask_b_low & (r <= 10) & ~mask_case1 & ~mask_case2 & ~mask_case3
        ec_case4 = 30.0 + g / 3.0
        ec_values = torch.where(mask_case4, ec_case4, ec_values)
        
        # Default case: EC = -1 for remaining cases
        mask_default = mask_b_low & ~mask_case1 & ~mask_case2 & ~mask_case3 & ~mask_case4
        ec_values = torch.where(mask_default, torch.full_like(ec_values, 0.0), ec_values)
        
        return ec_values
    
    def calculate_ec_from_coordinates_differentiable(self, lat_coords, lon_coords):
        """
        Calculate EC values from coordinates using fully differentiable operations
        
        Args:
            lat_coords: Tensor of latitude coordinates [batch_size]
            lon_coords: Tensor of longitude coordinates [batch_size]
            
        Returns:
            ec_values: Tensor of EC values [batch_size]
        """
        # Get RGB values
        rgb_values = self.get_rgb_from_coordinates_tensor(lat_coords, lon_coords)
        #print(rgb_values)
        # Convert to EC
        ec_values = self.rgb_to_ec_tensor(rgb_values)
        
        return ec_values
    
    def get_valid_coordinates_mask(self, lat_coords, lon_coords):
        """
        Create mask for valid coordinates
        
        Args:
            lat_coords: Tensor of latitude coordinates [batch_size]
            lon_coords: Tensor of longitude coordinates [batch_size]
            
        Returns:
            valid_mask: Boolean tensor [batch_size]
        """
        lat_valid = (lat_coords >= -90) & (lat_coords <= 90)
        lon_valid = (lon_coords >= -180) & (lon_coords <= 180)
        
        return lat_valid & lon_valid
    
    def calculate_midpoint_ec(self):
        """
        Calculate EC value for the point in the middle between the two reference points
        
        Returns:
            ec_value: EC value for the midpoint
            midpoint_coords: (lat, lon) coordinates of the midpoint
        """
        # Extract reference points
        x1, y1, lat1, lon1 = self.ref_points[0]
        x2, y2, lat2, lon2 = self.ref_points[1]
        
        # Calculate midpoint coordinates
        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0
        
        # Convert to tensors for the calculation
        mid_lat_tensor = torch.tensor([mid_lat], device=self.device, dtype=torch.float32)
        mid_lon_tensor = torch.tensor([mid_lon], device=self.device, dtype=torch.float32)
        
        # Calculate EC value using the differentiable function
        ec_value = self.calculate_ec_from_coordinates_differentiable(mid_lat_tensor, mid_lon_tensor)
        
        return ec_value.item(), (mid_lat, mid_lon)
    
    def print_reference_points_info(self):
        """
        Print information about the reference points and calculate midpoint EC
        """
        print("=== Reference Points Information ===")
        
        # Extract reference points
        x1, y1, lat1, lon1 = self.ref_points[0]
        x2, y2, lat2, lon2 = self.ref_points[1]
        
        print(f"Point 1: Pixel ({x1:.1f}, {y1:.1f}) → Lat/Lon ({lat1:.4f}, {lon1:.4f})")
        print(f"Point 2: Pixel ({x2:.1f}, {y2:.1f}) → Lat/Lon ({lat2:.4f}, {lon2:.4f})")
        
        # Calculate midpoint
        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0
        
        print(f"\nMidpoint: Lat/Lon ({mid_lat:.4f}, {mid_lon:.4f})")
        
        # Calculate EC value for midpoint
        ec_value, _ = self.calculate_midpoint_ec()
        print(f"Midpoint EC Value: {ec_value:.4f}")
        
        # Calculate pixel coordinates for midpoint
        mid_x, mid_y = self.lat_lon_to_xy_tensor(
            torch.tensor([mid_lat], device=self.device),
            torch.tensor([mid_lon], device=self.device)
        )
        print(f"Midpoint Pixel Coordinates: ({mid_x.item():.1f}, {mid_y.item():.1f})")
        
        return ec_value, (mid_lat, mid_lon)

    def compare_sampling_methods(self, lat_coords, lon_coords):
        """
        Compare direct indexing vs F.grid_sample interpolation
        
        Args:
            lat_coords: Tensor of latitude coordinates [batch_size]
            lon_coords: Tensor of longitude coordinates [batch_size]
            
        Returns:
            direct_rgb: RGB values from direct indexing
            interpolated_rgb: RGB values from F.grid_sample
        """
        if self.rgb_tensor is None:
            print("No RGB tensor loaded for comparison")
            return None, None
            
        # Convert to pixel coordinates
        x_coords, y_coords = self.lat_lon_to_xy_tensor(lat_coords, lon_coords)
        
        # Clamp coordinates to valid range
        height, width = self.rgb_tensor.shape[:2]
        x_coords = torch.clamp(x_coords, 0, width - 1)
        y_coords = torch.clamp(y_coords, 0, height - 1)
        
        # Method 1: Direct indexing (nearest neighbor)
        x_indices = torch.floor(x_coords).long()
        y_indices = torch.floor(y_coords).long()
        x_indices = torch.clamp(x_indices, 0, width - 1)
        y_indices = torch.clamp(y_indices, 0, height - 1)
        direct_rgb = self.rgb_tensor[y_indices, x_indices]
        
        # Method 2: F.grid_sample with bilinear interpolation
        x_norm = 2.0 * x_coords / (width - 1) - 1.0
        y_norm = 2.0 * y_coords / (height - 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1).unsqueeze(1)
        rgb_for_sampling = self.rgb_tensor.permute(2, 0, 1).unsqueeze(0)
        batch_size = lat_coords.shape[0]
        grid_expanded = grid.expand(batch_size, -1, -1, -1)
        
        sampled_rgb = F.grid_sample(
            rgb_for_sampling.expand(batch_size, -1, -1, -1),
            grid_expanded,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        interpolated_rgb = sampled_rgb.squeeze(-1).squeeze(-1)
        
        return direct_rgb, interpolated_rgb

def test_differentiability():
    """Test that the differentiable EC calculator preserves gradients"""
    print("=== Testing Differentiable EC Calculator ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculator = DifferentiableECCalculator(device=device)
    
    # Test midpoint calculation
    print("\n" + "="*50)
    ec_value, (mid_lat, mid_lon) = calculator.print_reference_points_info()
    print("="*50)
    
    # Create test data
    batch_size = 4
    lat_coords = torch.tensor([40.7128, 51.5074, 35.6762, -33.8688], 
                             device=device, requires_grad=True)
    lon_coords = torch.tensor([-74.0060, -0.1278, 139.6503, 151.2093], 
                             device=device, requires_grad=True)
    
    print(f"\nInput coordinates shape: {lat_coords.shape}")
    print(f"Requires grad: {lat_coords.requires_grad}")
    
    # Calculate EC values
    ec_values = calculator.calculate_ec_from_coordinates_differentiable(lat_coords, lon_coords)
    
    print(f"Output EC values: {ec_values}")
    print(f"EC values shape: {ec_values.shape}")
    print(f"EC values requires_grad: {ec_values.requires_grad}")
    
    # Test backward pass
    loss = ec_values.sum()
    loss.backward()
    
    print(f"Gradients computed: {lat_coords.grad is not None}")
    
    if lat_coords.grad is not None:
        print(f"Lat gradients: {lat_coords.grad}")
        print("✅ Gradient computation successful!")
    else:
        print("❌ Gradient computation failed!")
    
    return calculator

def test_sampling_comparison():
    """Test the difference between direct indexing and F.grid_sample"""
    print("\n=== Testing Sampling Methods Comparison ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculator = DifferentiableECCalculator(device=device)
    
    # Test with exact pixel coordinates (should be identical)
    print("\n1. Testing with exact pixel coordinates:")
    lat_coords = torch.tensor([36.6232, -3.3291], device=device)  # Reference points
    lon_coords = torch.tensor([11.1788, -39.1475], device=device)
    
    direct_rgb, interpolated_rgb = calculator.compare_sampling_methods(lat_coords, lon_coords)
    
    if direct_rgb is not None:
        print(f"Direct indexing RGB: {direct_rgb}")
        print(f"Interpolated RGB: {interpolated_rgb}")
        print(f"Are they identical? {torch.allclose(direct_rgb, interpolated_rgb, atol=1e-6)}")
    
    # Test with fractional coordinates (should be different)
    print("\n2. Testing with fractional coordinates:")
    # Use midpoint coordinates which will have fractional pixel values
    mid_lat = (36.6232 + (-3.3291)) / 2.0
    mid_lon = (11.1788 + (-39.1475)) / 2.0
    
    lat_coords = torch.tensor([mid_lat], device=device)
    lon_coords = torch.tensor([mid_lon], device=device)
    
    direct_rgb, interpolated_rgb = calculator.compare_sampling_methods(lat_coords, lon_coords)
    
    if direct_rgb is not None:
        print(f"Direct indexing RGB: {direct_rgb}")
        print(f"Interpolated RGB: {interpolated_rgb}")
        print(f"Are they identical? {torch.allclose(direct_rgb, interpolated_rgb, atol=1e-6)}")
        print(f"Difference: {torch.abs(direct_rgb - interpolated_rgb)}")
    
    # Test with random coordinates
    print("\n3. Testing with random coordinates:")
    lat_coords = torch.tensor([20.0, 30.0, 40.0], device=device)
    lon_coords = torch.tensor([-10.0, 0.0, 10.0], device=device)
    
    direct_rgb, interpolated_rgb = calculator.compare_sampling_methods(lat_coords, lon_coords)
    
    if direct_rgb is not None:
        print(f"Direct indexing RGB: {direct_rgb}")
        print(f"Interpolated RGB: {interpolated_rgb}")
        print(f"Are they identical? {torch.allclose(direct_rgb, interpolated_rgb, atol=1e-6)}")
        print(f"Max difference: {torch.max(torch.abs(direct_rgb - interpolated_rgb))}")
        print(f"Mean difference: {torch.mean(torch.abs(direct_rgb - interpolated_rgb))}")

if __name__ == "__main__":
    #test_differentiability()
    #test_sampling_comparison() 
    a=1