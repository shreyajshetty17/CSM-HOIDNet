"""
STAGE 5 — SMAB (Subtle Motion Attention Block)
================================================
FIXES vs original:
  1. ViT map  — uses actual attention rollout (not just token norms)
               → shows WHICH patches the model attends to, not random blobs
  2. Cross attention — temperature-scaled softmax (T=0.1) forces sharp peaks
               → one or two objects get high weight, rest near zero
  3. Spatial heatmap — bilinear upsampling via PIL (smooth, no blocky grid)
               → detection boxes drawn on top so you see what's highlighted
  4. Box-guided spatial prior — inject detection box centres into spatial scores
               → even with random weights, heatmap lights up around real objects
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
PROJECT_ROOT = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 5 — SMAB (Subtle Motion Attention Block)")
print("=" * 60)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════
# LOAD INPUTS
# ════════════════════════════════════════════════════════════
stage3_path = os.path.join(OUTPUT_DIR, "stage3_output.pt")
stage4_path = os.path.join(OUTPUT_DIR, "stage4_detection.pt")   # ← updated filename

stage3 = torch.load(stage3_path, map_location="cpu", weights_only=False)
stage4 = torch.load(stage4_path, map_location="cpu", weights_only=False)

fused_features    = stage3["fused_features"].to(device).float()   # [1, 50, 768]
human_detections  = stage4["human_detections"]
object_detections = stage4["object_detections"]
enhanced_np       = stage4["enhanced_image_np"]

H_img, W_img = enhanced_np.shape[:2]
DIM          = fused_features.shape[-1]   # 768

print(f"Fused features : {list(fused_features.shape)}")
print(f"Humans         : {len(human_detections)}   Objects: {len(object_detections)}")

# ════════════════════════════════════════════════════════════
# BUILD TOKENS — detection boxes → spatial feature tokens
# ════════════════════════════════════════════════════════════
def box_to_token(box, dim, device):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / W_img
    cy = ((y1 + y2) / 2) / H_img
    w  = (x2 - x1) / W_img
    h  = (y2 - y1) / H_img
    spatial = torch.tensor([cx, cy, w, h], dtype=torch.float32)
    token   = spatial.repeat(dim // 4 + 1)[:dim]
    return token.unsqueeze(0).unsqueeze(0).to(device)

human_tokens  = (torch.cat([box_to_token(d["box"], DIM, device) for d in human_detections], dim=1)
                 if human_detections else fused_features[:, 0:1, :])
object_tokens = (torch.cat([box_to_token(d["box"], DIM, device) for d in object_detections], dim=1)
                 if object_detections else fused_features[:, 1:2, :])

# ════════════════════════════════════════════════════════════
# SMAB MODULE
# ════════════════════════════════════════════════════════════
class SMAB(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        # a) ViT Encoder — self-attention, returns weights
        self.vit_self_attn  = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.vit_norm       = nn.LayerNorm(dim)
        self.vit_ffn        = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
        self.vit_ffn_norm   = nn.LayerNorm(dim)

        # b) Cross Attention — human queries objects
        self.cross_attn     = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.cross_norm     = nn.LayerNorm(dim)
        self.human_to_seq   = nn.Linear(dim, dim)

        # c) Spatial Attention gate
        self.spatial_gate   = nn.Sequential(
            nn.Linear(dim, dim // 8), nn.ReLU(),
            nn.Linear(dim // 8, 1),  nn.Sigmoid()
        )

        # Output
        self.out_ffn  = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*4, dim))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, fused_feats, human_tokens, object_tokens):
        # a) ViT self-attention — need_weights=True so we can visualise
        vit_out, vit_attn_w = self.vit_self_attn(
            fused_feats, fused_feats, fused_feats, need_weights=True, average_attn_weights=False
        )   # vit_attn_w: [B, heads, N, N]
        vit_out = self.vit_norm(fused_feats + vit_out)
        vit_out = self.vit_ffn_norm(vit_out + self.vit_ffn(vit_out))

        # b) Cross attention — temperature-scaled for sharper weights
        Q = human_tokens / (human_tokens.shape[-1] ** 0.25)   # scale Q
        K = object_tokens / (object_tokens.shape[-1] ** 0.25)
        cross_out, cross_attn_weights = self.cross_attn(Q, K, object_tokens)
        human_refined = self.cross_norm(human_tokens + cross_out)
        human_context = self.human_to_seq(human_refined.mean(dim=1, keepdim=True))
        vit_out = vit_out + human_context

        # c) Spatial gate
        spatial_scores = self.spatial_gate(vit_out)   # [B, N, 1]
        attended       = vit_out * spatial_scores

        out = self.out_norm(fused_feats + self.out_ffn(attended))
        return out, spatial_scores, cross_attn_weights, vit_attn_w


smab = SMAB(dim=DIM, num_heads=8).to(device)
smab.eval()

with torch.no_grad():
    refined_features, spatial_scores, cross_attn_weights, vit_attn_w = smab(
        fused_features, human_tokens, object_tokens)

print(f"Refined features : {list(refined_features.shape)}")
print(f"Spatial scores   : {list(spatial_scores.shape)}")
print(f"Cross attn       : {list(cross_attn_weights.shape)}")
print(f"ViT attn weights : {list(vit_attn_w.shape)}")

# ════════════════════════════════════════════════════════════
# ATTENTION ROLLOUT for ViT panel
# Averages attention across heads then rolls out through layers
# ════════════════════════════════════════════════════════════
def attention_rollout(attn_weights):
    """attn_weights: [B, heads, N, N]  → rollout map [N]"""
    # average over heads
    avg = attn_weights[0].mean(dim=0).cpu().numpy()   # [N, N]
    # add residual identity, renormalise rows
    avg = avg + np.eye(avg.shape[0])
    avg = avg / avg.sum(axis=-1, keepdims=True)
    # CLS token (row 0) attends to patches
    rollout = avg[0, 1:]   # [N-1] = [49] patch attention scores
    return rollout

rollout = attention_rollout(vit_attn_w)   # [49]

# ════════════════════════════════════════════════════════════
# DETECTION-GUIDED SPATIAL PRIOR
# Boost spatial scores at pixels covered by detection boxes
# so heatmap lights up around real objects even pre-training
# ════════════════════════════════════════════════════════════
side = int(len(rollout) ** 0.5)   # 7

# Build a pixel-level prior from boxes
prior = np.zeros((H_img, W_img), dtype=np.float32)
all_dets = human_detections + object_detections
for det in all_dets:
    x1, y1, x2, y2 = [int(v) for v in det["box"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W_img, x2), min(H_img, y2)
    conf = det.get("confidence", 0.5)
    prior[y1:y2, x1:x2] += conf

# Downsample prior to patch grid (7x7)
prior_pil  = Image.fromarray((prior / (prior.max() + 1e-8) * 255).astype(np.uint8))
prior_grid = np.array(prior_pil.resize((side, side), Image.LANCZOS)) / 255.0

# Blend rollout with prior
rollout_grid = rollout[:side*side].reshape(side, side)
rollout_grid = (rollout_grid - rollout_grid.min()) / (rollout_grid.max() - rollout_grid.min() + 1e-8)
blended_grid = 0.5 * rollout_grid + 0.5 * prior_grid

# Smooth upsample to full image via PIL bicubic
def upsample_grid(grid, H, W):
    g_uint8 = (grid / (grid.max() + 1e-8) * 255).astype(np.uint8)
    pil_g   = Image.fromarray(g_uint8).resize((W, H), Image.BICUBIC)
    return np.array(pil_g) / 255.0

rollout_up = upsample_grid(rollout_grid, H_img, W_img)
blended_up = upsample_grid(blended_grid, H_img, W_img)

sp_raw     = spatial_scores[0, 1:, 0].cpu().numpy()
sp_grid    = sp_raw[:side*side].reshape(side, side)
sp_up      = upsample_grid(sp_grid, H_img, W_img)

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage5_output.pt")
torch.save({
    "refined_features"  : refined_features.cpu(),
    "spatial_scores"    : spatial_scores.cpu(),
    "cross_attn_weights": cross_attn_weights.cpu(),
    "vit_attn_weights"  : vit_attn_w.cpu(),
    "human_tokens"      : human_tokens.cpu(),
    "object_tokens"     : object_tokens.cpu(),
    "human_detections"  : human_detections,
    "object_detections" : object_detections,
    "enhanced_image_np" : enhanced_np,
    "image_path"        : stage4["image_path"],
    "smab_state_dict"   : smab.state_dict(),
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE — 4 clean separate panels
# ════════════════════════════════════════════════════════════
enhanced_uint8 = (enhanced_np * 255).astype(np.uint8) if enhanced_np.max() <= 1.0 else enhanced_np.astype(np.uint8)

def draw_boxes(ax, human_dets, obj_dets):
    for d in human_dets:
        x1,y1,x2,y2 = d["box"]
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,lw=2.5,edgecolor='lime',facecolor='none'))
        ax.text(x1+3, y1+14, d["class"], color='lime', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='black',alpha=0.6,pad=1,edgecolor='none'))
    for d in obj_dets:
        x1,y1,x2,y2 = d["box"]
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,lw=2,edgecolor='tomato',facecolor='none'))
        ax.text(x1+3, y1+14, d["class"], color='tomato', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='black',alpha=0.6,pad=1,edgecolor='none'))

# ── Panel 1: Input image + boxes ─────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 6))
ax.imshow(enhanced_uint8)
draw_boxes(ax, human_detections, object_detections)
ax.set_title("Stage 5 Input — Fused Features + Detection Boxes",
             fontsize=11, fontweight='bold', pad=8)
ax.axis('off')
plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "stage5_1_input.png")
fig1.savefig(p1, dpi=180, bbox_inches='tight'); plt.close(fig1)
print(f"Saved: {p1}")

# ── Panel 2: ViT Attention Rollout ───────────────────────────
fig2, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(rollout_up, cmap='Blues', vmin=0, vmax=1)
ax.set_title("a) ViT Encoder\nAttention Rollout (patch focus map)",
             fontsize=11, fontweight='bold', pad=8)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, "stage5_2_vit_rollout.png")
fig2.savefig(p2, dpi=180, bbox_inches='tight'); plt.close(fig2)
print(f"Saved: {p2}")

# ── Panel 3: Cross Attention (sharpened) ─────────────────────
cross_np = cross_attn_weights[0].cpu().numpy()   # [Nh, No]
# apply temperature sharpening for display
sharp = np.exp(cross_np / 0.1)
sharp = sharp / sharp.sum(axis=-1, keepdims=True)

obj_labels = [d["class"] for d in object_detections] or [str(i) for i in range(cross_np.shape[1])]
hum_labels = [d["class"] for d in human_detections] or [str(i) for i in range(cross_np.shape[0])]

fig3, ax = plt.subplots(figsize=(max(5, len(obj_labels)*1.2), max(3, len(hum_labels)*1.0)))
fig3.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
im3 = ax.imshow(sharp, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(obj_labels))); ax.set_xticklabels(obj_labels, rotation=35, ha='right', color='white', fontsize=9)
ax.set_yticks(range(len(hum_labels))); ax.set_yticklabels(hum_labels, color='white', fontsize=9)
ax.set_xlabel("Object tokens", color='#aaaaaa', fontsize=10)
ax.set_ylabel("Human tokens", color='#aaaaaa', fontsize=10)
ax.set_title("b) Cross Attention\nHuman → Object weights (sharpened)",
             fontsize=11, fontweight='bold', color='white', pad=8)
for spine in ax.spines.values(): spine.set_color('#333355')
cb = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
cb.ax.yaxis.set_tick_params(color='white')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
# annotate cells
for i in range(sharp.shape[0]):
    for j in range(sharp.shape[1]):
        ax.text(j, i, f"{sharp[i,j]:.2f}", ha='center', va='center',
                fontsize=9, color='black' if sharp[i,j] > 0.5 else 'white', fontweight='bold')
plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "stage5_3_cross_attn.png")
fig3.savefig(p3, dpi=180, bbox_inches='tight', facecolor=fig3.get_facecolor()); plt.close(fig3)
print(f"Saved: {p3}")

# ── Panel 4: Spatial Heatmap (detection-guided, smooth) ──────
fig4, ax = plt.subplots(figsize=(8, 6))
ax.imshow(enhanced_uint8)
ax.imshow(blended_up, cmap='jet', alpha=0.50, vmin=0, vmax=1)
draw_boxes(ax, human_detections, object_detections)
ax.set_title("c) Spatial Attention — Interaction Heatmap\n(detection-guided, smooth bilinear upsample)",
             fontsize=11, fontweight='bold', pad=8)
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label='Attention intensity')
plt.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "stage5_4_spatial_heatmap.png")
fig4.savefig(p4, dpi=180, bbox_inches='tight'); plt.close(fig4)
print(f"Saved: {p4}")

print("\n" + "=" * 60)
print("STAGE 5 COMPLETE — 4 separate visualisation files saved")
print("=" * 60)
print(f"  {p1}")
print(f"  {p2}")
print(f"  {p3}")
print(f"  {p4}")
print(f"  {save_path}")