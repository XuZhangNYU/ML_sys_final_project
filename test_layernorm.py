import torch
import torch.nn as nn
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# We need to find the 'build' folder. 
# Since your script is in `kernel_code/src/test_case.py`, 
# and build is likely in `BigDataMLSys/build`, we need to go up two levels.
# We will check both ../build and ../../build just to be safe.
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


try:
    import bten
    print("✅ 'bten' CUDA extension imported successfully.")
except ImportError:
    print("❌ ERROR: Could not import 'bten'.")
    exit(1)

# ---------------------------------------------------------
# 1. GPT-2 Specific LayerNorm (Reference)
# ---------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# ---------------------------------------------------------
# 2. Test Harness
# ---------------------------------------------------------
def test_layernorm():
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    # 1. Setup Dimensions
    # GPT-2 Small: Hidden=768
    B, S, H = 2, 32, 768
    eps = 1e-5
    
    print(f"Testing LayerNorm with shape [{B}, {S}, {H}]...")

    # 2. PyTorch Reference (Using GPT2 implementation)
    ln_ref = LayerNorm(H, eps=eps).to(device)
    
    # Create Input
    x_torch = torch.randn(B, S, H, device=device)
    
    # Run PyTorch
    y_ref = ln_ref(x_torch)

    # 3. Custom Bten Implementation
    # Convert Input
    x_bten = bten.TensorF(B, 1, S, H, True) # [B, 1, S, D] structure for GPT-2
    x_bten.copy_from_numpy(x_torch.detach().cpu().numpy())

    # Convert Weights (Gamma) & Bias (Beta)
    # Reshape to [1, H] for broadcast logic in your kernel
    gamma_np = ln_ref.weight.detach().cpu().numpy().reshape(1, H)
    beta_np  = ln_ref.bias.detach().cpu().numpy().reshape(1, H)
    
    gamma_bten = bten.TensorF(1, H, True)
    gamma_bten.copy_from_numpy(gamma_np)
    
    beta_bten = bten.TensorF(1, H, True)
    beta_bten.copy_from_numpy(beta_np)

    # Run Custom Op
    y_bten = x_bten.layernorm(gamma_bten, beta_bten, eps)

    # 4. Compare
    y_bten_np = y_bten.to_numpy().reshape(B, S, H)
    y_ref_np = y_ref.detach().cpu().numpy()

    diff = np.abs(y_ref_np - y_bten_np).max()
    print(f"Max Difference: {diff:.8f}")

    if diff < 1e-4:
        print("✅ SUCCESS: LayerNorm matches GPT-2 Implementation!")
    else:
        print("❌ FAILURE: Outputs diverge.")
        print("Ref sample:", y_ref_np[0, 0, :5])
        print("Bten sample:", y_bten_np[0, 0, :5])

if __name__ == "__main__":
    test_layernorm()