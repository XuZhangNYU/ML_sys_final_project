import torch
import torch.nn as nn
import math
import numpy as np
import copy

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
# 0. Setup Custom CUDA Extension
# ---------------------------------------------------------
try:
    import bten
    print("✅ 'bten' CUDA extension imported successfully.")
except ImportError:
    print("❌ ERROR: Could not import 'bten'.")
    exit(1)

class GPT2Config:
    def __init__(self, n_embd=768, n_head=12, n_ctx=1024):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_ctx = n_ctx

# ---------------------------------------------------------
# 1. The ORIGINAL GPT-2 Implementation (Conv1D)
# ---------------------------------------------------------
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

class OriginalAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(OriginalAttention, self).__init__()
        n_state = nx 
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

# ---------------------------------------------------------
# 2. Your Custom Implementation (nn.Linear + bten)
# ---------------------------------------------------------
class CustomAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        self.n_head = config.n_head
        self.split_size = nx
        self.scale = scale
        self.c_attn = nn.Linear(nx, nx * 3)
        self.c_proj = nn.Linear(nx, nx)
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def _to_torch(self, t_bten, original_shape, device):
        # We can now export 4D numpy directly if you updated bindings,
        # but let's stick to safe flatten/view to be sure.
        x_np = t_bten.to_numpy().flatten()
        return torch.from_numpy(x_np).view(*original_shape).to(device)

    def _attn(self, q, k, v):
        bs, heads, seq_len, head_dim = q.size()
        
        # 1. Prepare Q: [B, H, S, D]
        # We MUST use the 4D constructor so C++ knows b=bs, d=heads
        q_cont = q.contiguous()
        q_bten = bten.TensorF(bs, heads, seq_len, head_dim, True)
        q_bten.copy_from_numpy(q_cont.detach().cpu().numpy())

        # 2. Prepare K: [B, H, D, S]
        # Transposed in memory, so we construct it as (B, H, D, S)
        k_cont = k.contiguous()
        k_bten = bten.TensorF(bs, heads, head_dim, seq_len, True)
        k_bten.copy_from_numpy(k_cont.detach().cpu().numpy())
        
        # 3. BMM: Q @ K
        # C++ uses internal metadata to calculate batch_count = bs * heads
        w_bten = q_bten.bmm(k_bten)
        
        # Convert back for masking (Python side)
        w = self._to_torch(w_bten, (bs, heads, seq_len, seq_len), q.device)

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # 4. Softmax (Custom Kernel)
        # We can treat this as a 2D operation [Rows, Cols]
        # But to be clean, let's pass it as 4D [B, H, S, S]
        w_cont = w.contiguous()
        w_softmax_in = bten.TensorF(bs, heads, seq_len, seq_len, True)
        w_softmax_in.copy_from_numpy(w_cont.detach().cpu().numpy())
        
        w_softmax_out = w_softmax_in.softmax()
        
        # 5. W @ V
        # W_out: [B, H, S, S]
        # V: [B, H, S, D]
        v_cont = v.contiguous()
        v_bten = bten.TensorF(bs, heads, seq_len, head_dim, True)
        v_bten.copy_from_numpy(v_cont.detach().cpu().numpy())
        
        # BMM Output: [B, H, S, D]
        out_bten = w_softmax_out.bmm(v_bten)
        
        return self._to_torch(out_bten, (bs, heads, seq_len, head_dim), q.device)

    # ... (Keep forward, split_heads, merge_heads same as before) ...
    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k: return x.permute(0, 2, 3, 1)
        else: return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

# ---------------------------------------------------------
# 3. Test Harness
# ---------------------------------------------------------
def compare_models():
    torch.manual_seed(42)
    device = torch.device("cuda")
    
    # 1. Setup
    C = GPT2Config(n_embd=128, n_head=4, n_ctx=64)
    seq_len = 32
    batch_size = 2
    
    print(f"Comparison: Batch={batch_size}, Seq={seq_len}, HeadDim={128//4}")

    # 2. Instantiate
    orig_model = OriginalAttention(C.n_embd, C.n_ctx, C, scale=True).to(device)
    custom_model = CustomAttention(C.n_embd, C.n_ctx, C, scale=True).to(device)

    # 3. TRANSPOSE & SYNC WEIGHTS (The Critical Step)
    # Conv1D weights: [n_inputs, n_outputs]
    # Linear weights: [n_outputs, n_inputs]
    with torch.no_grad():
        # Transpose Weight, Copy Bias
        custom_model.c_attn.weight.data = orig_model.c_attn.weight.t().contiguous()
        custom_model.c_attn.bias.data = orig_model.c_attn.bias.clone()
        
        custom_model.c_proj.weight.data = orig_model.c_proj.weight.t().contiguous()
        custom_model.c_proj.bias.data = orig_model.c_proj.bias.clone()

    # 4. Run
    x = torch.randn(batch_size, seq_len, C.n_embd, device=device)
    
    orig_out = orig_model(x)
    custom_out = custom_model(x)

    # 5. Check
    diff = (orig_out - custom_out).abs().max().item()
    print(f"Max Difference: {diff:.8f}")
    
    if torch.allclose(orig_out, custom_out, atol=1e-4, rtol=1e-4):
        print("✅ SUCCESS: Custom Attention matches Original GPT-2 Logic!")
    else:
        print("❌ FAILURE: Mismatch detected.")

if __name__ == "__main__":
    compare_models()