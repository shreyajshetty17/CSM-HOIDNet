"""
STAGE 3 — ATTENTION-BASED MULTI-SCALE FUSION
==============================================
Architecture role : 4 CLIP scale features -> Fused Feature
Input  : 4 x [1, 50, 768] from Stage 2 (outputs/stage2_visualization.png)
Output : fused [1, 50, 768] saved to outputs/stage3_output.pt

Method : finest-scale features query all 4 scales via cross-attention,
         followed by residual + FFN refinement.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
PROJECT_ROOT = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 3 — ATTENTION-BASED MULTI-SCALE FUSION")
print("=" * 60)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════
# LOAD STAGE 2 OUTPUT
# ════════════════════════════════════════════════════════════
stage2_path = os.path.join(OUTPUT_DIR, "stage2_output.pt")
if not os.path.exists(stage2_path):
    raise FileNotFoundError(
        f"{stage2_path} not found.\nRun 2_clip.py first.")

stage2 = torch.load(
    stage2_path,
    map_location="cpu",
    weights_only=False
)
scale_features = [f.to(device).float() for f in stage2["scale_features"]]
tap_layers = stage2["tap_layers"]

print(f"\nLoaded Stage 2 output : {stage2_path}")
for layer, feat in zip(tap_layers, scale_features):
    print(f"  Layer {layer:2d} -> shape {list(feat.shape)}")

# ════════════════════════════════════════════════════════════
# MULTI-SCALE FUSION MODULE
# ════════════════════════════════════════════════════════════
class MultiScaleFusion(nn.Module):
    def __init__(self, dim=768, num_scales=4, num_heads=8):
        super().__init__()
        # Project each scale to same dim before attention
        self.scale_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])
        # Cross-attention: finest scale (query) attends to all scales (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, scale_features):
        projected = [proj(f) for proj, f in zip(self.scale_proj, scale_features)]
        query = projected[-1]                  # finest scale = query  [B,50,768]
        keys  = torch.cat(projected, dim=1)    # all scales concat    [B,200,768]
        attended, attn_weights = self.cross_attn(query, keys, keys)
        x = self.norm1(query + attended)
        out = self.norm2(x + self.ffn(x))
        return out, attn_weights               # [B,50,768], [B,50,200]


model = MultiScaleFusion(dim=768, num_scales=4, num_heads=8).to(device)
model.eval()
print("\n✓ MultiScaleFusion module initialised")
print("  NOTE: this module is UNTRAINED here - weights are random.")
print("  It gets trained later in the full training loop (Stage 9+).")

# ════════════════════════════════════════════════════════════
# RUN FUSION
# ════════════════════════════════════════════════════════════
with torch.no_grad():
    fused_features, attn_weights = model(scale_features)

print(f"\n===== FUSION RESULTS =====")
print(f"Input  : 4 x {list(scale_features[0].shape)}")
print(f"Output : {list(fused_features.shape)}")
print(f"Attn   : {list(attn_weights.shape)}")

f = fused_features.detach().cpu()
print(f"\nFused feature stats:")
print(f"  mean = {f.mean().item():.4f}")
print(f"  std  = {f.std().item():.4f}")
print(f"  norm = {torch.norm(f).item():.4f}")

# Per-scale attention contribution
N = scale_features[0].shape[1]   # 50 tokens per scale
print(f"\nAttention contribution per scale:")
for i, layer in enumerate(tap_layers):
    block_attn = attn_weights[:, :, i*N:(i+1)*N].mean().item()
    print(f"  Layer {layer:2d} -> mean attention = {block_attn:.4f}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage3_output.pt")
torch.save({
    "fused_features": fused_features.cpu(),   # [1, 50, 768]
    "attn_weights"  : attn_weights.cpu(),
    "scale_features": [s.cpu() for s in scale_features],
    "tap_layers"    : tap_layers,
    "enhanced_image_np": stage2["enhanced_image_np"],
    "image_path"    : stage2["image_path"],
    "fusion_state_dict": model.state_dict(),
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

contributions = [attn_weights[:, :, i*N:(i+1)*N].mean().item()
                  for i in range(len(tap_layers))]
bars = axes[0].bar([f"Layer {l}" for l in tap_layers], contributions,
                   color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
axes[0].set_title("Stage 3: Attention Weight\nper Scale (untrained)", fontweight='bold')
axes[0].set_ylabel("Mean attention weight")
for bar, val in zip(bars, contributions):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                  f"{val:.4f}", ha='center', fontsize=8)

patch_tokens = (
    fused_features[0, 1:, :]
    .mean(dim=-1)
    .detach()
    .cpu()
    .numpy()
)
side = int(len(patch_tokens) ** 0.5)
heatmap = patch_tokens[:side*side].reshape(side, side)
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
im = axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
axes[1].set_title("Stage 3: Fused Feature\nSpatial Map (7x7 patches)", fontweight='bold')
plt.colorbar(im, ax=axes[1], fraction=0.046)
axes[1].axis('off')

plt.suptitle("Stage 3 - Multi-Scale Fusion", fontsize=13, fontweight='bold')
plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage3_visualization.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {viz_path}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"Input  : 4 x {list(scale_features[0].shape)}")
print(f"Output : {list(fused_features.shape)} (fused features)")
