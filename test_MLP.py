import torch
import torch.nn as nn
import math
import numpy as np

import sys
import os

# --- STEP 1: SETUP PATHS (MUST BE FIRST) ---
# Get the directory of this script (src)
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

if not found:
    print(f"CRITICAL WARNING: Could not find 'build' directory in {possible_paths}")

# ---------------------------------------------------------
# 0. Imports & Setup
# ---------------------------------------------------------
try:
    import bten
    print("✅ 'bten' CUDA extension imported successfully.")
except ImportError:
    print("❌ ERROR: Could not import 'bten'.")
    exit(1)

class GPT2Config:
    def __init__(self, n_embd=768, n_inner=None):
        self.n_embd = n_embd
        # GPT-2 default: Inner dim is 4 * embedding dim
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd

# ---------------------------------------------------------
# 1. The ORIGINAL GPT-2 MLP (Reference)
# ---------------------------------------------------------
def gelu_python(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class ReferenceMLP(nn.Module):
    def __init__(self, n_state, config):
        super(ReferenceMLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu_python

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

# ---------------------------------------------------------
# 2. Your Custom MLP (Target)
# ---------------------------------------------------------
class CustomMLP(nn.Module):
    def __init__(self, n_state, config): 
        super().__init__()
        nx = config.n_embd
        # We use nn.Linear instead of Conv1D
        self.c_fc = nn.Linear(nx, n_state)
        self.c_proj = nn.Linear(n_state, nx)
        # Note: We don't assign self.act here because we call .gelu() on the tensor

    def _to_bten(self, x_torch):
        # Flatten to 2D for MLP operations (Batch*Seq, Dim)
        flat = x_torch.contiguous().view(-1, x_torch.shape[-1])
        t = bten.TensorF(flat.shape[0], flat.shape[1], True)
        t.copy_from_numpy(flat.detach().cpu().numpy())
        return t

    def _to_torch(self, t_bten, shape, device):
        # Convert back and reshape to 3D [Batch, Seq, Dim]
        return torch.from_numpy(t_bten.to_numpy()).view(*shape).to(device)

    def forward(self, x):
        # x shape: [Batch, Seq, Dim]
        
        # 1. Convert Input
        x_bten = self._to_bten(x)
        
        # 2. Linear Layer 1 (c_fc) -> Expansion
        # Conv1D weights are [In, Out]. Linear weights are [Out, In].
        # For x @ W, we need [In, Out].
        # So we take Linear.weight.t() to get [In, Out].
        w1 = self._to_bten(self.c_fc.weight.t().contiguous()) 
        b1 = self._to_bten(self.c_fc.bias)
        
        # Matrix Mult: [N, In] @ [In, Out] + [Out]
        # Bten supports operator overloading for @ and +
        h = x_bten @ w1 + b1
        
        # 3. Activation (GELU) using Custom Kernel
        h = h.gelu()
        
        # 4. Linear Layer 2 (c_proj) -> Projection
        w2 = self._to_bten(self.c_proj.weight.t().contiguous())
        b2 = self._to_bten(self.c_proj.bias)
        
        h2 = h @ w2 + b2
        
        return self._to_torch(h2, x.shape, x.device)


class BtenMLP:
    def __init__(self, n_state, config):
        self.nx = config.n_embd
        self.n_state = n_state
        self.c_fc_w = None
        self.c_fc_b = None
        self.c_proj_w = None
        self.c_proj_b = None

    def import_weights(self, pytorch_mlp_module):
        with torch.no_grad():
            # Check source type (Conv1D vs Linear)
            # Conv1D weights are [In, Out]. Linear are [Out, In].
            
            if isinstance(pytorch_mlp_module.c_fc, Conv1D):
                # Conv1D is ALREADY [In, Out]. No Transpose needed!
                print("Importing from Conv1D (Original GPT2 style)...")
                w1 = pytorch_mlp_module.c_fc.weight.contiguous().detach().cpu().numpy()
                w2 = pytorch_mlp_module.c_proj.weight.contiguous().detach().cpu().numpy()
            elif isinstance(pytorch_mlp_module.c_fc, nn.Linear):
                # Linear is [Out, In]. TRANSPOSE REQUIRED.
                print("Importing from nn.Linear...")
                w1 = pytorch_mlp_module.c_fc.weight.t().contiguous().detach().cpu().numpy()
                w2 = pytorch_mlp_module.c_proj.weight.t().contiguous().detach().cpu().numpy()
            else:
                raise ValueError("Unknown layer type")

            b1 = pytorch_mlp_module.c_fc.bias.contiguous().detach().cpu().numpy()
            b2 = pytorch_mlp_module.c_proj.bias.contiguous().detach().cpu().numpy()

        print(f"  Weights W1 Shape: {w1.shape} (Expected: {self.nx}, {self.n_state})")
        print(f"  Weights W2 Shape: {w2.shape} (Expected: {self.n_state}, {self.nx})")

        # Load into Bten
        self.c_fc_w = bten.TensorF(1, 1, w1.shape[0], w1.shape[1], True)
        self.c_fc_w.copy_from_numpy(w1)
        
        self.c_fc_b = bten.TensorF(1, 1, 1, b1.shape[0], True)
        self.c_fc_b.copy_from_numpy(b1.reshape(1, -1))

        self.c_proj_w = bten.TensorF(1, 1, w2.shape[0], w2.shape[1], True)
        self.c_proj_w.copy_from_numpy(w2)
        
        self.c_proj_b = bten.TensorF(1, 1, 1, b2.shape[0], True)
        self.c_proj_b.copy_from_numpy(b2.reshape(1, -1))

    def forward(self, x):
        # x input is [Batch, 1, Seq, Dim] or similar
        # We flatten to [Batch*Seq, Dim]
        rows = x.shape[0] * x.shape[1] * x.shape[2]
        dim = x.shape[3]
        
        # Validation
        if dim != self.nx:
            raise RuntimeError(f"Input dim {dim} != Model dim {self.nx}")

        x_flat = x.view(1, 1, rows, dim)
        
        # FC
        h = x_flat @ self.c_fc_w
        h = h + self.c_fc_b 
        
        # Act
        h = h.gelu()
        
        # Proj
        h2 = h @ self.c_proj_w
        h2 = h2 + self.c_proj_b
        
        # Restore shape
        return h2.view(x.shape[0], x.shape[1], x.shape[2], self.nx)



# ---------------------------------------------------------
# 3. Verification
# ---------------------------------------------------------
def test_mlp():
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    # 1. Configuration
    n_embd = 768
    n_inner = 4 * n_embd # 3072
    config = GPT2Config(n_embd=n_embd, n_inner=n_inner)
    
    # Dimensions
    batch_size = 2
    seq_len = 32
    
    print(f"Testing MLP: [Batch={batch_size}, Seq={seq_len}, Embd={n_embd}] -> Inner={n_inner}")

    # 2. Instantiate Models
    ref_model = ReferenceMLP(n_inner, config).to(device)
    custom_model = BtenMLP(n_inner, config)

    # ---------------------------------------------------------
    # 3. WEIGHT LOADING & TRANSPOSTION (The Critical Part)
    # ---------------------------------------------------------
    # This simulates loading a GPT-2 checkpoint.
    # Original GPT-2 weights (Conv1D) are [Input, Output].
    # PyTorch nn.Linear weights are [Output, Input].
    # Therefore, we must TRANSPOSE (.t()) when loading into CustomMLP.
    
    print("Syncing weights (Simulating GPT-2 Checkpoint Load)...")
    # with torch.no_grad():
    #     # c_fc: Expand
    #     # Ref (Conv1D): [768, 3072]
    #     # Custom (Linear): Expects [3072, 768]
    #     custom_model.c_fc.weight.data = ref_model.c_fc.weight.data.t().contiguous()
    #     custom_model.c_fc.bias.data = ref_model.c_fc.bias.data.clone()
        
    #     # c_proj: Project back
    #     # Ref (Conv1D): [3072, 768]
    #     # Custom (Linear): Expects [768, 3072]
    #     custom_model.c_proj.weight.data = ref_model.c_proj.weight.data.t().contiguous()
    #     custom_model.c_proj.bias.data = ref_model.c_proj.bias.data.clone()
    custom_model.import_weights(ref_model)
    # 4. Generate Data
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, n_embd).cuda()
    x_bten = bten.TensorF(batch_size, 1, seq_len, n_embd, True)
    x_bten.copy_from_numpy(x.detach().cpu().numpy())
    # 5. Run Inference
    print("Running Reference MLP...")
    y_ref = ref_model(x)
    
    print("Running Custom MLP...")
    y_custom = custom_model.forward(x_bten)

    # 6. Compare
    print("\n--- Results ---")
    


    out_bten_np = y_custom.to_numpy().reshape(2, 32, n_embd)
    diff = np.abs(y_ref.detach().cpu().numpy() - out_bten_np).max()
    print(f"MLP Max Diff: {diff:.6f}")

if __name__ == "__main__":
    test_mlp()