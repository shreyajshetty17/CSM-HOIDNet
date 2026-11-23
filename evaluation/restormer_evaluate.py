import os
import cv2
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# ==========================================================
# PATHS
# ==========================================================

ORIGINAL_IMAGE = r"C:\Users\STUDENT\Downloads\Shreya Internship\Dataset\hico_20160224_det\hico_20160224_det\images\test2015\HICO_test2015_00009068.jpg"

ENHANCED_IMAGE = r"C:\Users\STUDENT\Downloads\hoid_project\outputs\stage1_enhanced.png"

# ==========================================================
# LOAD IMAGES
# ==========================================================

original = cv2.imread(ORIGINAL_IMAGE)
enhanced = cv2.imread(ENHANCED_IMAGE)

if original is None:
    raise FileNotFoundError(
        f"Original image not found:\n{ORIGINAL_IMAGE}"
    )

if enhanced is None:
    raise FileNotFoundError(
        f"Enhanced image not found:\n{ENHANCED_IMAGE}"
    )

original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

# ==========================================================
# ENSURE SAME SIZE
# ==========================================================

if original.shape != enhanced.shape:
    enhanced = cv2.resize(
        enhanced,
        (original.shape[1], original.shape[0])
    )

# ==========================================================
# METRICS
# ==========================================================

mse = np.mean(
    (original.astype(np.float32) -
     enhanced.astype(np.float32)) ** 2
)

mae = np.mean(
    np.abs(
        original.astype(np.float32) -
        enhanced.astype(np.float32)
    )
)

psnr = peak_signal_noise_ratio(
    original,
    enhanced,
    data_range=255
)

ssim = structural_similarity(
    original,
    enhanced,
    channel_axis=2,
    data_range=255
)

# ==========================================================
# IMAGE SIZE INFO
# ==========================================================

orig_h, orig_w = original.shape[:2]
enh_h, enh_w = enhanced.shape[:2]

orig_pixels = orig_h * orig_w
enh_pixels = enh_h * enh_w

orig_size_kb = os.path.getsize(ORIGINAL_IMAGE) / 1024
enh_size_kb = os.path.getsize(ENHANCED_IMAGE) / 1024

# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("RESTORMER EVALUATION REPORT")
print("=" * 70)

print("\nIMAGE RESOLUTION")
print("-" * 70)
print(f"Original : {orig_w} x {orig_h}")
print(f"Enhanced : {enh_w} x {enh_h}")

print("\nPIXEL COUNT")
print("-" * 70)
print(f"Original : {orig_pixels:,}")
print(f"Enhanced : {enh_pixels:,}")

print("\nFILE SIZE")
print("-" * 70)
print(f"Original : {orig_size_kb:.2f} KB")
print(f"Enhanced : {enh_size_kb:.2f} KB")

print("\nQUALITY METRICS")
print("-" * 70)
print(f"PSNR : {psnr:.4f} dB")
print(f"SSIM : {ssim:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")

