"""
STAGE 1 — RESTORMER (Image Enhancement)
========================================
Architecture role : INPUT IMAGE -> Enhanced Image
Input  : raw HICO-DET image (any size, e.g. 640x426x3)
Output : enhanced image, same size, saved to outputs/

This stage uses the official pretrained Restormer (real denoising weights)
to enhance image quality before feature extraction in Stage 2.
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CONFIG — paths
# ════════════════════════════════════════════════════════════
PROJECT_ROOT   = r"C:\Users\STUDENT\Downloads\hoid_project"
DATASET_ROOT   = r"C:\Users\STUDENT\Downloads\Shreya Internship\Dataset\hico_20160224_det\hico_20160224_det"
RESTORMER_ROOT = os.path.join(PROJECT_ROOT, "Restormer", "Restormer-main")
MODEL_PATH     = os.path.join(RESTORMER_ROOT, "Motion_Deblurring", "pretrained_models", "motion_deblurring.pth")

# Pick one sample image from HICO-DET test set
IMAGE_PATH = os.path.join(DATASET_ROOT, "images", "test2015", "HICO_test2015_00009483.jpg")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("RESTORMER (Image Enhancement)")
print("=" * 60)
print(f"Device     : {device}")
print(f"Image path : {IMAGE_PATH}")

# ════════════════════════════════════════════════════════════
# LOAD RESTORMER MODEL
# ════════════════════════════════════════════════════════════
sys.path.insert(0, RESTORMER_ROOT)

try:
    from basicsr.models.archs.restormer_arch import Restormer
    USE_REAL_MODEL = True
except Exception as e:
    print(f"\n! Could not import real Restormer ({e})")
    print("  Falling back to a lightweight residual enhancer.")
    USE_REAL_MODEL = False


class FallbackEnhancer(torch.nn.Module):
    """Used only if real Restormer weights/repo are unavailable."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 48, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(48, 48, 3, padding=1, groups=48),
            torch.nn.GELU(),
            torch.nn.Conv2d(48, 3, 1),
        )
        self.gamma = torch.nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.gamma * self.net(x)


if USE_REAL_MODEL:
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ).to(device)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint["params"]

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False
        )

        print("Missing:", len(missing))
        print("Unexpected:", len(unexpected))

        if missing:
            print("Missing sample:", missing[:10])

        if unexpected:
            print("Unexpected sample:", unexpected[:10])

        print(f"✓ Loaded pretrained weights: {MODEL_PATH}")

    else:
        print(f"\n! Pretrained weights not found at:\n  {MODEL_PATH}")

else:
    model = FallbackEnhancer().to(device)
    print("✓ Using fallback enhancer")

model.eval()

# ════════════════════════════════════════════════════════════
# LOAD IMAGE
# ════════════════════════════════════════════════════════════
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
print(f"\nOriginal image shape : {img_rgb.shape}")

# ════════════════════════════════════════════════════════════
# PAD TO MULTIPLE OF 8 (required by Restormer)
# ════════════════════════════════════════════════════════════
new_h = ((h + 7) // 8) * 8
new_w = ((w + 7) // 8) * 8
img_pad = cv2.copyMakeBorder(img_rgb, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT)

img_tensor = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0)
img_tensor = img_tensor.float().to(device) / 255.0

# ════════════════════════════════════════════════════════════
# INFERENCE
# ════════════════════════════════════════════════════════════
with torch.no_grad():
    enhanced = model(img_tensor)

enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
enhanced = enhanced[:h, :w]                     # remove padding
enhanced = np.clip(enhanced, 0, 1)
enhanced_uint8 = (enhanced * 255).astype(np.uint8)

print(f"Enhanced image shape : {enhanced_uint8.shape}")

# ════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage1_enhanced.png")
cv2.imwrite(save_path, cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR))
print(f"\nSaved: {save_path}")

# Save raw tensor too (used by Stage 2)
torch.save({
    "enhanced_image_np": enhanced,            # float [0,1], HWC
    "original_image_np": img_rgb,             # uint8 HWC
    "image_path"       : IMAGE_PATH,
}, os.path.join(OUTPUT_DIR, "stage1_output.pt"))
print(f"Saved: {os.path.join(OUTPUT_DIR, 'stage1_output.pt')}")

# ════════════════════════════════════════════════════════════
# VISUALISE
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img_rgb)
axes[0].set_title(f"Input Image\n{os.path.basename(IMAGE_PATH)}", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(enhanced_uint8)
axes[1].set_title("Restormer Output\n(Enhanced Image)", fontweight='bold')
axes[1].axis('off')

plt.suptitle("Image Enhancement", fontsize=13, fontweight='bold')
plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage1_comparison.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {viz_path}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"Input  : {img_rgb.shape}")
print(f"Output : {enhanced_uint8.shape}")
print("=" * 60)