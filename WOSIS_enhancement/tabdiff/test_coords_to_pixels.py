from differentiable_ec_calculator import DifferentiableECCalculator

import torch


def main():
    calculator=DifferentiableECCalculator()

    # Test the calculator
    lat_coords = torch.tensor([22.5135],device='cuda')
    lon_coords = torch.tensor([-109.3151],device='cuda')
    pixels = calculator.lat_lon_to_xy_tensor(lat_coords, lon_coords)
    print(pixels)
    
    
    return



if __name__ == '__main__':
    main()

