import os
import torch
import numpy as np

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

stage2_path = os.path.join(OUTPUT_DIR, "stage2_output.pt")

print("=" * 70)
print("CLIP FEATURE EVALUATION")
print("=" * 70)

data = torch.load(
    stage2_path,
    map_location="cpu",
    weights_only=False
)

scale_features = data["scale_features"]
global_feature = data["global_feature"]

# ============================================================
# GLOBAL FEATURE
# ============================================================
print("\nGLOBAL FEATURE")
print("-" * 70)

g = global_feature.float()

print("Shape      :", list(g.shape))
print("Mean       :", g.mean().item())
print("Std Dev    :", g.std().item())
print("Min        :", g.min().item())
print("Max        :", g.max().item())
print("L2 Norm    :", torch.norm(g).item())

# ============================================================
# MULTI-SCALE FEATURES
# ============================================================
print("\nMULTI-SCALE FEATURES")
print("-" * 70)

for idx, feat in enumerate(scale_features):

    feat = feat.float()

    print(f"\nLayer {idx+1}")

    print("Shape      :", list(feat.shape))
    print("Mean       :", feat.mean().item())
    print("Std Dev    :", feat.std().item())
    print("Min        :", feat.min().item())
    print("Max        :", feat.max().item())
    print("L2 Norm    :", torch.norm(feat).item())

# ============================================================
# OVERALL ANALYSIS
# ============================================================
all_feats = torch.cat(
    [f.reshape(-1, f.shape[-1]) for f in scale_features],
    dim=0
)

print("\n" + "=" * 70)
print("OVERALL FEATURE STATISTICS")
print("=" * 70)

print("Combined Shape :", list(all_feats.shape))
print("Mean           :", all_feats.mean().item())
print("Std Dev        :", all_feats.std().item())
print("Min            :", all_feats.min().item())
print("Max            :", all_feats.max().item())
print("L2 Norm        :", torch.norm(all_feats).item())

print("\nEvaluation Complete")