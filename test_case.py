from __future__ import annotations
import sys
import os

# --- STEP 1: SETUP PATHS (MUST BE FIRST) ---
# Get the directory of this script (src)
script_dir = os.path.dirname(os.path.abspath(__file__))

# find build manually
possible_paths = [
    os.path.join(script_dir, "build"),       # Check kernel_code/build
    os.path.join(script_dir, "..", "build")  # Check BigDataMLSys/build
]

found = False
for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        if abs_path not in sys.path:
            sys.path.append(abs_path)
        print(f"Found build directory at: {abs_path}")
        print(f"Contents: {os.listdir(abs_path)}")
        found = True
        break

if not found:
    print(f"CRITICAL WARNING: Could not find 'build' directory in {possible_paths}")

# --- STEP 2: NOW IMPORT LIBRARIES ---
import torch
import numpy as np
from typing import Optional, Union
from contextlib import contextmanager

# Now that sys.path is fixed, this import will work
try:
    import bten
    print("SUCCESS: 'bten' module imported successfully!")
except ImportError as e:
    print(f"FAIL: Still cannot import 'bten'. Error: {e}")
    sys.exit(1)
    

def test_4d_bmm_correctness():
    print("\n=== Test: 4D Batched Matrix Multiplication (BMM) ===")
    
    B = 2    # Batch
    D = 4    # Heads (Depth)
    S = 32   # Sequence Length
    H = 64   # Head Dimension
    
    print(f"Config: Batch={B}, Heads={D}, Seq={S}, HeadDim={H}")

    # 2. Create Random Data
    # Q: [Batch, Heads, Seq, HeadDim]
    q_np = np.random.randn(B, D, S, H).astype(np.float32)
    # K: [Batch, Heads, HeadDim, Seq] (Transposed for Attention)
    k_np = np.random.randn(B, D, H, S).astype(np.float32)

    # 3. Run PyTorch Reference
    q_torch = torch.from_numpy(q_np).cuda()
    k_torch = torch.from_numpy(k_np).cuda()
    
    # [B, D, S, H] @ [B, D, H, S] -> [B, D, S, S]
    out_torch = torch.matmul(q_torch, k_torch)
    
    # Initialize 4D Tensors directly from Numpy
    q_bten = bten.TensorF(B, D, S, H, True)
    q_bten.copy_from_numpy(q_np)
    
    k_bten = bten.TensorF(B, D, H, S, True)
    k_bten.copy_from_numpy(k_np)
    
    # The C++ object knows its own shape.
    out_bten = q_bten.bmm(k_bten)
    
    out_bten_np = out_bten.to_numpy() # Should return 4D array [B, D, S, S]
    out_torch_np = out_torch.cpu().numpy()

    print(f"PyTorch Output Shape: {out_torch_np.shape}")
    print(f"Bten Output Shape:    {out_bten_np.shape}")

    # Check Shape
    if out_bten_np.shape != out_torch_np.shape:
        print("❌ SHAPE MISMATCH!")
        return

    # Check Values
    diff = np.abs(out_torch_np - out_bten_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Max Absolute Diff: {max_diff:.6f}")
    print(f"Mean Absolute Diff: {mean_diff:.6f}")

    if np.allclose(out_torch_np, out_bten_np, atol=1e-4, rtol=1e-4):
        print("v SUCCESS: BMM results match PyTorch!")
    else:
        print("Nah!!DS FAILURE: BMM results diverge.")
        
        # Debugging: Check where the error is
        # If Batch 0 is correct but Batch 1 is wrong, your strides are broken.
        for b in range(B):
            d_batch = np.abs(out_torch_np[b] - out_bten_np[b]).max()
            print(f"  Batch {b} Max Diff: {d_batch:.6f}")

def test_tensor_construction():
    print("\n=== Test: 4D Tensor Construction & Shapes ===")
    
    # 1. Create a 4D tensor
    B, D, H, W = 2, 3, 4, 5
    t = bten.TensorF(B, D, H, W, True)
    
    # 2. Check reported shape
    # This relies on you updating the .def_property_readonly("shape") in py_tensor_shim.hh
    shape = t.shape
    print(f"Created Tensor({B}, {D}, {H}, {W})")
    print(f"Reported Shape: {shape}")
    
    if len(shape) == 4 and shape == (B, D, H, W):
        print("v Shape reported correctly as 4D")
    elif len(shape) == 2 and shape == (B*D*H, W):
        print("warning  Shape reported as Flattened 2D (Legacy View). This is valid logic but harder to debug.")
    else:
        print(f"x Weird shape reported: {shape}")

    # 3. Test Round Trip
    data = np.arange(B*D*H*W, dtype=np.float32).reshape(B, D, H, W)
    t.copy_from_numpy(data)
    
    back = t.to_numpy()
    
    if np.allclose(data, back):
        print("v Data Round-Trip (Numpy -> GPU -> Numpy) Successful")
    else:
        print("x Data corruption during copy")

if __name__ == "__main__":
    try:
        test_tensor_construction()
        test_4d_bmm_correctness()
    except Exception as e:
        print(f"\n x CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()