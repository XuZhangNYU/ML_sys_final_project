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

# manually find build
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
    print("pass! 'bten' CUDA extension imported successfully.")
except ImportError:
    print(" ERROR: Could not import 'bten'.")
    exit(1)

class GPT2Config:
    def __init__(self, n_embd=768, n_head=12, n_ctx=1024):
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_ctx = n_ctx

# --------------------------------------
# 1. The ORIGINAL GPT-2 Implementation (Conv1D)

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


class BtenAttention:
    def __init__(self, nx, n_ctx, config, scale=False):
        self.n_head = config.n_head
        self.split_size = nx
        self.scale = scale
        self.head_dim = nx // self.n_head
        self.nx = nx
        
        # Weights (Initialized in import_weights)
        self.c_attn_w = None; self.c_attn_b = None
        self.c_proj_w = None; self.c_proj_b = None
        
        # Bias Mask (Pre-loaded)
        bias_np = np.tril(np.ones((1, 1, n_ctx, n_ctx), dtype=np.float32))
        self.bias_mask = bten.TensorF(1, 1, n_ctx, n_ctx, True)
        self.bias_mask.copy_from_numpy(bias_np)

    def import_weights(self, torch_attn_module):
        """Robust weight import handling Conv1D (No Transpose) vs Linear (Transpose)."""
        with torch.no_grad():
            # 1. c_attn
            if isinstance(torch_attn_module.c_attn, Conv1D):
                print("Importing c_attn from Conv1D (Direct copy)...")
                w1 = torch_attn_module.c_attn.weight.contiguous().cpu().numpy()
            else:
                print("Importing c_attn from Linear (Transpose)...")
                w1 = torch_attn_module.c_attn.weight.t().contiguous().cpu().numpy()
            b1 = torch_attn_module.c_attn.bias.contiguous().cpu().numpy()

            # 2. c_proj
            if isinstance(torch_attn_module.c_proj, Conv1D):
                print("Importing c_proj from Conv1D (Direct copy)...")
                w2 = torch_attn_module.c_proj.weight.contiguous().cpu().numpy()
            else:
                print("Importing c_proj from Linear (Transpose)...")
                w2 = torch_attn_module.c_proj.weight.t().contiguous().cpu().numpy()
            b2 = torch_attn_module.c_proj.bias.contiguous().cpu().numpy()

        # Load to Bten
        self.c_attn_w = bten.TensorF(1, 1, w1.shape[0], w1.shape[1], True)
        self.c_attn_w.copy_from_numpy(w1)
        self.c_attn_b = bten.TensorF(1, 1, 1, b1.shape[0], True)
        self.c_attn_b.copy_from_numpy(b1.reshape(1,-1))

        self.c_proj_w = bten.TensorF(1, 1, w2.shape[0], w2.shape[1], True)
        self.c_proj_w.copy_from_numpy(w2)
        self.c_proj_b = bten.TensorF(1, 1, 1, b2.shape[0], True)
        self.c_proj_b.copy_from_numpy(b2.reshape(1,-1))

    def forward(self, x):
        B = x.shape[0]; S = x.shape[2]; Dim = x.shape[3]
        
        # 1. QKV Proj (Linear)
        # Flatten x to [B*S, Dim] for Linear
        x_flat = x.view(1, 1, B*S, Dim)

        qkv = x_flat @ self.c_attn_w + self.c_attn_b
        # qkv: [B*S, 3*Dim]
        
        # 1. Split
        # Now q, k, v are [B*S, Dim]
        
        # 2. View as [B, Seq, Heads, HeadDim]
        B = x.shape[0]; S = x.shape[2]

        # 3. Permute [B, S, H, D] -> [B, H, S, D]
        q, k, v = qkv.split_qkv(self.n_head, self.head_dim)

        # 2. Reshape to 4D [B, S, H, D]
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_head, self.head_dim)
        v = v.view(B, S, self.n_head, self.head_dim)

        # 3. Permute
        q = q.permute_0213() # [B, H, S, D]
        k = k.permute_0231() # [B, H, D, S] <--- Used 0231!
        v = v.permute_0213() # [B, H, S, D]

        # 4. Attention
        # print(q.shape, k.shape, v.shape)
        w = q.bmm(k) # [B, H, S, S]
        # print("w", w.shape)
      
        w = w.causal_mask(scale=1.0/math.sqrt(self.head_dim))
        w = w.softmax()

        # print("w", w.shape)
        # print("v", v.shape)
        a = w.bmm(v) # [B, H, S, D]
        
        # 5. Merge (Permute Back)
        # [B, H, S, D] -> [B, S, H, D]
        a = a.permute_0213() # Need inverse kernel
        
        # Flatten
        a_flat = a.view(1, 1, B*S, self.nx)
        print("aflat", a_flat.shape)
        print("cprojw", self.c_proj_w.shape)

        out = a_flat.bmm(self.c_proj_w) + self.c_proj_b
        return out.view_3d(B, S, self.nx)

def _to_torch_3d(t_bten, device):
    # t_bten is [B, 1, S, D]
    arr_4d = t_bten.to_numpy()
    # Numpy reshape [B, 1, S, D] -> [B, S, D]
    arr_3d = arr_4d.reshape(arr_4d.shape[0], arr_4d.shape[2], arr_4d.shape[3])
    return torch.from_numpy(arr_3d).to(device)

# ---------------------------------------------------------
# 3. Test Harness
def compare_models():
    torch.manual_seed(42)
    
    n_embd = 768
    config = GPT2Config(n_embd=n_embd, n_head=12, n_ctx=1024)
    
    # 1. Models
    ref_model = OriginalAttention(n_embd, 1024, config, scale=True).cuda().eval()
    my_model = BtenAttention(n_embd, 1024, config, scale=True)
    
    # 2. Import Weights
    # This will trigger "Importing from Conv1D"
    my_model.import_weights(ref_model)
    
    # 3. Input
    x_torch = torch.randn(2, 32, n_embd).cuda()
    
    # 4. Run Ref
    y_ref = ref_model(x_torch)
    
    # 5. Run Custom
    x_bten = bten.TensorF(2, 1, 32, n_embd, True)
    x_bten.copy_from_numpy(x_torch.detach().cpu().numpy())
    
    y_bten = my_model.forward(x_bten)
    print("ref vs bten")
    print(y_ref.shape)
    print(y_bten.to_numpy().shape)
    # 6. Compare
    y_bten_np = y_bten.to_numpy()
    # y_bten_np = y_bten.to_numpy().reshape(2, 32, n_embd)

    y_ref_np = y_ref.detach().cpu().numpy()
    
    diff = np.abs(y_ref_np - y_bten_np).max() 
    # + np.abs(bb.to_numpy() - b.detach().cpu().numpy()).max() + np.abs(cc.to_numpy() - c.detach().cpu().numpy()).max()
    print(f"\nMax Difference: {diff:.8f}")
    
    if diff < 1e-4:
        print("v SUCCESS!")
    else:
        print("x FAILED")

if __name__ == "__main__":
    compare_models()