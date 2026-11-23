import os
import torch
import numpy as np

print("=" * 60)
print("STAGE 3 EVALUATION — MULTI-SCALE FUSION")
print("=" * 60)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

fusion_path = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "stage3_output.pt"    
)

# --------------------------------------------------
# Load
# --------------------------------------------------

print("Loading:", fusion_path)
print("Exists :", os.path.exists(fusion_path))

data = torch.load(
    fusion_path,
    map_location="cpu",
    weights_only=False
)

print(f"\nLoaded: {fusion_path}")

print("\nData type:", type(data))

if isinstance(data, dict):
    print("\nAvailable keys:")
    for k in data.keys():
        print("  ", k)
# --------------------------------------------------
# Extract tensors
# --------------------------------------------------

fused_features = data["fused_features"]
attention = data["attn_weights"]

print("\n===== SHAPES =====")

print("Fused Features :", list(fused_features.shape))
print("Attention      :", list(attention.shape))

# --------------------------------------------------
# Feature Statistics
# --------------------------------------------------

feat = fused_features.detach().cpu().numpy()

print("\n===== FUSED FEATURE STATISTICS =====")

print(f"Mean     : {feat.mean():.6f}")
print(f"Std Dev  : {feat.std():.6f}")
print(f"Min      : {feat.min():.6f}")
print(f"Max      : {feat.max():.6f}")
print(f"L2 Norm  : {np.linalg.norm(feat):.6f}")

# --------------------------------------------------
# Attention Statistics
# --------------------------------------------------

attn = attention.detach().cpu().numpy()

print("\n===== ATTENTION STATISTICS =====")

print(f"Mean     : {attn.mean():.6f}")
print(f"Std Dev  : {attn.std():.6f}")
print(f"Min      : {attn.min():.6f}")
print(f"Max      : {attn.max():.6f}")

# --------------------------------------------------
# NaN / Inf Check
# --------------------------------------------------

print("\n===== NUMERICAL CHECK =====")

has_nan_feat = np.isnan(feat).any()
has_inf_feat = np.isinf(feat).any()

has_nan_attn = np.isnan(attn).any()
has_inf_attn = np.isinf(attn).any()

print("Fused Features:")
print("  NaN :", has_nan_feat)
print("  Inf :", has_inf_feat)

print("\nAttention:")
print("  NaN :", has_nan_attn)
print("  Inf :", has_inf_attn)

# --------------------------------------------------
# Attention per Scale
# --------------------------------------------------

print("\n===== ATTENTION PER SCALE =====")

num_scales = 4

tokens_per_scale = attn.shape[-1] // num_scales

for i, layer in enumerate([3, 6, 9, 11]):

    start = i * tokens_per_scale
    end = (i + 1) * tokens_per_scale

    scale_attn = attn[..., start:end]

    print(
        f"Layer {layer:2d} -> "
        f"mean={scale_attn.mean():.6f}, "
        f"std={scale_attn.std():.6f}"
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)