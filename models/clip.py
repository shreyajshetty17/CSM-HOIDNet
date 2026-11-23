"""
STAGE 2 — CLIP VISION BACKBONE (Feature Extraction)
====================================================
Architecture role : Enhanced Image -> Multi-scale CLIP features
Input  : enhanced image from Stage 1 (outputs/stage1_output.pt)
Output : 4 feature maps, each [1, 50, 768] (ViT layers 3, 6, 9, 11)
         saved to outputs/stage1_enhanced.png
"""

import os
import torch
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
print("STAGE 2 — CLIP VISION BACKBONE (Feature Extraction)")
print("=" * 60)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════
# LOAD STAGE 1 OUTPUT
# ════════════════════════════════════════════════════════════
stage1_path = os.path.join(OUTPUT_DIR, "stage1_output.pt")
if not os.path.exists(stage1_path):
    raise FileNotFoundError(
        f"{stage1_path} not found.\nRun 1_restormer.py first.")

stage1 = torch.load(
    stage1_path,
    map_location="cpu",
    weights_only=False
)
enhanced_np = stage1["enhanced_image_np"]   # float [0,1], HWC
print(f"\nLoaded Stage 1 output : {stage1_path}")
print(f"Enhanced image shape  : {enhanced_np.shape}")

# ════════════════════════════════════════════════════════════
# LOAD CLIP MODEL (open_clip, pretrained OpenAI weights)
# ════════════════════════════════════════════════════════════
import open_clip
import torch.nn.functional as F

print("\nLoading CLIP ViT-B-32 (OpenAI pretrained) ...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()
print("✓ CLIP model loaded")

# ════════════════════════════════════════════════════════════
# PREPROCESS ENHANCED IMAGE FOR CLIP (resize to 224x224)
# ════════════════════════════════════════════════════════════
from PIL import Image
enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
pil_img = Image.fromarray(enhanced_uint8)

image_input = preprocess(pil_img).unsqueeze(0).to(device)   # [1, 3, 224, 224]
print(f"CLIP input tensor shape : {list(image_input.shape)}")

# ════════════════════════════════════════════════════════════
# EXTRACT MULTI-SCALE FEATURES
# Tap ViT transformer blocks at layers 3, 6, 9, 11
# Each gives [1, 50, 768]  (1 CLS token + 49 patch tokens)
# ════════════════════════════════════════════════════════════
scale_features = []
tap_layers = [3, 6, 9, 11]

def make_hook(store):
    def hook(module, input, output):
        store.append(output.detach().cpu())
    return hook

handles = []
for i in tap_layers:
    h = model.visual.transformer.resblocks[i].register_forward_hook(
        make_hook(scale_features))
    handles.append(h)

with torch.no_grad():
    global_feature = model.encode_image(image_input)
    global_feature = global_feature / global_feature.norm(dim=-1, keepdim=True)

for h in handles:
    h.remove()

print(f"\n===== EXTRACTION RESULTS =====")
print(f"Global feature shape : {list(global_feature.shape)}")
print(f"Number of scale maps : {len(scale_features)}")
for layer, feat in zip(tap_layers, scale_features):
    print(f"  Layer {layer:2d} -> shape {list(feat.shape)}  "
          f"norm={feat.norm().item():.2f}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage2_output.pt")
torch.save({
    "scale_features" : scale_features,     # 4 x [1, 50, 768]
    "global_feature" : global_feature.cpu(),  # [1, 512]
    "tap_layers"      : tap_layers,
    "enhanced_image_np": enhanced_np,
    "image_path"      : stage1["image_path"],
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE — feature norms per layer
# ════════════════════════════════════════════════════════════
norms = [f.norm(dim=-1).mean().item() for f in scale_features]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(enhanced_uint8)
axes[0].set_title("Stage 2 Input\n(Enhanced Image from Stage 1)", fontweight='bold')
axes[0].axis('off')

bars = axes[1].bar([f"Layer {l}" for l in tap_layers], norms,
                   color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
axes[1].set_title("CLIP Multi-Scale Feature Norms\n(Stage 2 Output)", fontweight='bold')
axes[1].set_ylabel("Mean L2 norm")
for bar, val in zip(bars, norms):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f"{val:.1f}", ha='center', fontsize=9)

plt.suptitle("Stage 2 — CLIP Vision Backbone", fontsize=13, fontweight='bold')
plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage2_visualization.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {viz_path}")

print("\n" + "=" * 60)
print("STAGE 2 COMPLETE")
print("=" * 60)
print(f"Input  : enhanced image {enhanced_np.shape}")
print(f"Output : 4 x {list(scale_features[0].shape)} (CLIP multi-scale features)")
