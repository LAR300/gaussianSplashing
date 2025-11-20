#!/usr/bin/env python3
import sys
import torch
import traceback

def main():
    print("=== Simple HYB Test ===")
    try:
        from scene import GaussianModel
        from utils.loss_utils import loss_func
        
        # Test 1: Create GaussianModel
        print("Creating GaussianModel with HYB...")
        gaussians = GaussianModel(sh_degree=3, uw_flag="HYB")
        print("✓ GaussianModel created successfully")
        
        # Test 2: Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test 3: Simple tensor operations
        test_tensor = torch.tensor([[0.15, 0.20, 0.25]], device=device)
        print(f"✓ Test tensor created: {test_tensor}")
        
        print("All tests passed!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
