"""
STAGE 7 — SEMI-SUPERVISED CLIP-GUIDED CONTRASTIVE LEARNING
============================================================
Matches architecture diagram EXACTLY:

a) Data Preparation     - Labeled HOI triplets (Human,Action,Object)
                          + Unlabeled raw images (pseudo-labeled via CLIP)
b) Transformer Encoder  - Encodes HOI visual representations
c) CLIP Text Encoder    - Encodes text descriptions (frozen)
d) Contrastive Learning - Aligns image-text pairs in a BATCH
                          (diagonal matrix = positives, off-diagonal = negatives)

This demonstrates the contrastive learning module with a small batch
of samples so the similarity MATRIX (as shown in your diagram) is
visible, not just a single 1x4 comparison.
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
print("STAGE 7 — Semi-Supervised CLIP-Guided Contrastive Learning")
print("=" * 60)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════
# LOAD STAGE 5 + 6 OUTPUT
# ════════════════════════════════════════════════════════════
stage5_path = os.path.join(OUTPUT_DIR, "stage5_output.pt")
stage6_path = os.path.join(OUTPUT_DIR, "stage6_output.pt")

if not os.path.exists(stage5_path):
    raise FileNotFoundError(f"{stage5_path} not found. Run 5_smab.py first.")
if not os.path.exists(stage6_path):
    raise FileNotFoundError(f"{stage6_path} not found. Run 6_eacrm.py first.")

stage5 = torch.load(stage5_path, map_location="cpu", weights_only=False)
stage6 = torch.load(stage6_path, map_location="cpu", weights_only=False)

refined_features  = stage5["refined_features"].to(device).float()  # [1, 50, 768]
human_detections   = stage6["human_detections"]
object_detections  = stage6["object_detections"]
enhanced_np         = stage6["enhanced_image_np"]
llm_data            = stage6.get("llm_data", {})

print(f"\nLoaded Stage 5 — refined features : {list(refined_features.shape)}")
print(f"Loaded Stage 6 — humans={len(human_detections)}  objects={len(object_detections)}")

# ════════════════════════════════════════════════════════════
# a) DATA PREPARATION
#    LABELED triplet : built from Stage 6's real detections + reasoning
#                       e.g. (Person, riding, horse)
#    UNLABELED data   : synthetic variations (simulating "raw images
#                       without ground truth") -> CLIP pseudo-labels them
#
#    This matches the diagram: Labeled Data (Human,Action,Object) +
#    Unlabeled Data (Raw Images) feeding into the SAME encoder.
# ════════════════════════════════════════════════════════════
print("\na) Data Preparation")
print("-" * 50)

# FIX: sort by confidence, pick highest-confidence object (not random set order)
sorted_objects = sorted(object_detections, key=lambda d: d["confidence"], reverse=True)
obj_classes = list(dict.fromkeys(d["class"] for d in sorted_objects)) or ["object"]
print(f"  Objects sorted by confidence: {[(d['class'], round(d['confidence'],3)) for d in sorted_objects]}")
main_object = obj_classes[0]
action      = llm_data.get("primary_action", f"interacting with a {main_object}")
human_label = "Person" if human_detections else "Entity"

# ---- LABELED triplet (ground-truth-style, from real pipeline output) ----
labeled_triplet = (human_label, action.split()[0] if action else "interacting", main_object)
labeled_text     = f"a {human_label.lower()} {action}"
print(f"  LABELED triplet   : {labeled_triplet}")
print(f"  LABELED text      : '{labeled_text}'")

# ---- UNLABELED candidates (simulating raw/unannotated images) ----
# In real semi-supervised training these would be OTHER unlabeled images
# in the dataset. Here we simulate plausible/implausible variants for the
# SAME image so CLIP must pseudo-label them (assign confidence).
unlabeled_candidates = [
    f"a {human_label.lower()} {action}",                     # matches labeled (positive)
    f"a {human_label.lower()} eating a {main_object}",        # implausible negative
    f"a {human_label.lower()} carrying a {main_object}",      # plausible-ish negative
    f"an empty scene with no people",                         # hard negative
    f"a {human_label.lower()} sleeping",                      # unrelated negative
]
print(f"\n  UNLABELED candidates (to be pseudo-labeled by CLIP):")
for i, t in enumerate(unlabeled_candidates):
    tag = "[matches labeled]" if i == 0 else "[unlabeled]"
    print(f"    {i}: {t}  {tag}")

all_texts = unlabeled_candidates
N = len(all_texts)

# ════════════════════════════════════════════════════════════
# b) TRANSFORMER ENCODER
#    Encode the SAME refined visual features N times (simulating a
#    batch where this one labeled sample is compared against all
#    unlabeled candidates) -> produces the visual side of the matrix
# ════════════════════════════════════════════════════════════
print("\nb) Transformer Encoder")
print("-" * 50)

class HOITransformerEncoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads,
            dim_feedforward=in_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.proj(x)

hoi_encoder = HOITransformerEncoder(in_dim=768, out_dim=512).to(device)
hoi_encoder.eval()
print("  HOI Transformer Encoder initialised (untrained)")

with torch.no_grad():
    hoi_representation = hoi_encoder(refined_features)          # [1, 512]
    hoi_representation = F.normalize(hoi_representation, dim=-1)
    # Repeat to simulate batch position (same visual sample, compared to N texts)
    visual_batch = hoi_representation.repeat(N, 1)               # [N, 512]

print(f"  HOI representation : {list(hoi_representation.shape)}")
print(f"  Visual batch (repeated for matrix view) : {list(visual_batch.shape)}")

# ════════════════════════════════════════════════════════════
# c) CLIP TEXT ENCODER (frozen)
# ════════════════════════════════════════════════════════════
print("\nc) CLIP Text Encoder (frozen)")
print("-" * 50)

import open_clip
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device).eval()
for p in clip_model.parameters():
    p.requires_grad = False

tokenizer = open_clip.get_tokenizer('ViT-B-32')
tokens    = tokenizer(all_texts).to(device)

with torch.no_grad():
    text_features = clip_model.encode_text(tokens)
    text_features = F.normalize(text_features.float(), dim=-1)   # [N, 512]

print(f"  CLIP text features : {list(text_features.shape)}")

# ════════════════════════════════════════════════════════════
# d) CONTRASTIVE LEARNING (InfoNCE, batch-style matrix)
#    Pull positives (index 0, matching labeled triplet) closer,
#    push all other (unlabeled/negative) pairs apart.
# ════════════════════════════════════════════════════════════
print("\nd) Contrastive Learning")
print("-" * 50)

temperature   = 0.07
sim_matrix    = (visual_batch @ text_features.T) / temperature    # [N, N]
# Only the diagonal element [0,0] is the TRUE positive pair in this demo
# (single labeled sample matched against its own correct text).
similarities_to_positive = sim_matrix[0]                          # [N] - row 0
probs = F.softmax(similarities_to_positive, dim=-1)

labels = torch.tensor([0], device=device)
contrastive_loss = F.cross_entropy(similarities_to_positive.unsqueeze(0), labels)

print(f"  Similarity matrix shape : {list(sim_matrix.shape)}")
print(f"  Probabilities (row 0)   : {probs.cpu().detach().numpy().round(4)}")
print(f"  Contrastive loss        : {contrastive_loss.item():.4f}")
print(f"\n  NOTE: loss is high/random because HOI encoder is UNTRAINED.")
print(f"  During real multi-epoch training (10_train.py), this loss")
print(f"  decreases as the model learns to align visual <-> text HOI pairs.")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage7_output.pt")
torch.save({
    "hoi_representation" : hoi_representation.cpu(),
    "text_features"      : text_features.cpu(),
    "sim_matrix"          : sim_matrix.cpu(),
    "all_texts"           : all_texts,
    "labeled_triplet"     : labeled_triplet,
    "contrastive_loss"    : contrastive_loss.item(),
    "refined_features"    : stage5["refined_features"],
    "human_detections"    : human_detections,
    "object_detections"   : object_detections,
    "enhanced_image_np"   : enhanced_np,
    "hoi_encoder_state_dict": hoi_encoder.state_dict(),
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE — matches diagram's 4-panel layout + similarity matrix
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1 - input image
axes[0].imshow(enhanced_np)
axes[0].set_title(f"a) Data Preparation\nLabeled triplet: {labeled_triplet}",
                   fontweight='bold', fontsize=10)
axes[0].axis('off')

# Panel 2 - contrastive bar chart (row 0 of matrix)
probs_np = probs.cpu().detach().numpy()
colors   = ['green'] + ['lightgray'] * (N - 1)
short_texts = [t[:32] for t in all_texts]
bars = axes[1].barh(range(N), probs_np, color=colors)
axes[1].set_yticks(range(N))
axes[1].set_yticklabels(short_texts, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel("Contrastive probability")
axes[1].set_title(f"d) Contrastive Learning\n(labeled vs unlabeled candidates, loss={contrastive_loss.item():.3f})",
                   fontweight='bold', fontsize=10)
for i, p in enumerate(probs_np):
    axes[1].text(p + 0.01, i, f"{p:.3f}", va='center', fontsize=8)

# Panel 3 - full similarity matrix (matches diagram's colored grid)
im = axes[2].imshow(sim_matrix.cpu().detach().numpy(), cmap='Blues', aspect='auto')
axes[2].set_title("Similarity Matrix\n(diagonal = positives, off-diag = negatives)",
                   fontweight='bold', fontsize=10)
axes[2].set_xlabel("Text candidates")
axes[2].set_ylabel("Visual samples (repeated)")
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.suptitle("Stage 7 — Semi-Supervised CLIP-Guided Contrastive Learning",
             fontsize=13, fontweight='bold')
plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage7_visualization.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {viz_path}")

print("\n" + "=" * 60)
print("STAGE 7 COMPLETE")
print("=" * 60)
print(f"Labeled triplet     : {labeled_triplet}")
print(f"HOI representation  : {list(hoi_representation.shape)}")
print(f"Contrastive loss     : {contrastive_loss.item():.4f}  (untrained)")
print("Next -> Stage 8: HOI Transformer Decoder")
print("=" * 60)