"""
STAGE 8 — HOI TRANSFORMER DECODER (Final Output)
==================================================
Architecture role : Refined features -> (Human, Action, Object) triplet
Input  : refined features (outputs/stage5_output.pt)
         + detections + reasoning (outputs/stage6_output.pt, stage7_output.pt)
Output : final HOI triplet prediction
         saved to outputs/stage8_output.pt

Components:
  a) Object Queries          - learnable queries for Human, Action, Object
  b) Transformer Decoder Layers - attend to encoded features + queries
  c) Prediction Heads        - predict Human, Action, Object for each query

This completes the full 8-stage pipeline for ONE sample image.
"""

import os
import torch
import torch.nn as nn
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

# HICO-DET has 117 verb classes and 80 object classes (COCO)
NUM_VERBS   = 117
NUM_OBJECTS = 80
NUM_QUERIES = 16   # small number of HOI queries for this demo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 8 — HOI Transformer Decoder (Final Output)")
print("=" * 60)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════
# LOAD STAGE 5 OUTPUT (refined features = decoder memory)
# ════════════════════════════════════════════════════════════
stage5_path = os.path.join(OUTPUT_DIR, "stage5_output.pt")
if not os.path.exists(stage5_path):
    raise FileNotFoundError(
        f"{stage5_path} not found.\nRun 5_smab.py first.")

stage5 = torch.load(
    stage5_path,
    map_location="cpu",
    weights_only=False
)
refined_features = stage5["refined_features"].to(device).float()  # [1, 50, 768]
print(f"\nLoaded Stage 5 output : {stage5_path}")
print(f"Refined features (decoder memory) : {list(refined_features.shape)}")

# ════════════════════════════════════════════════════════════
# LOAD STAGE 6/7 OUTPUT (for context + visualisation)
# ════════════════════════════════════════════════════════════
stage6_path = os.path.join(OUTPUT_DIR, "stage6_output.pt")
stage7_path = os.path.join(OUTPUT_DIR, "stage7_output.pt")

if os.path.exists(stage6_path):
    stage6 = torch.load(
        stage6_path,
        map_location="cpu",
        weights_only=False
    )
else:
    stage6 = None
if os.path.exists(stage7_path):
    stage7 = torch.load(
        stage7_path,
        map_location="cpu",
        weights_only=False
    )
else:
    stage7 = None
if stage6 is None:
    raise FileNotFoundError(f"{stage6_path} not found.\nRun 6_eacrm.py first.")

human_detections   = stage6["human_detections"]
object_detections  = stage6["object_detections"]
enhanced_np         = stage6["enhanced_image_np"]
reasoning_output    = stage6["reasoning_output"]

print(f"Loaded Stage 6 output : {stage6_path}")
print(f"  Reasoning: {reasoning_output[:60]}...")

# COCO class names for object prediction interpretation
from transformers import DetrForObjectDetection
_tmp = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
COCO_CLASSES = _tmp.config.id2label
del _tmp

# HICO-DET verb list (117 verbs, standard order)
HICO_VERBS = [
    "adjust","assemble","block","blow","board","break","brush_with","buy",
    "carry","catch","chase","check","clean","control","cook","cut",
    "cut_with","direct","drag","dribble","drink_with","drive","dry","eat",
    "eat_at","exit","feed","fill","flip","flush","fly","greet","grind",
    "groom","herd","hit","hold","hop_on","hose","hug","hunt","inspect",
    "install","jump","kick","kiss","lasso","launch","lick","lie_on","lift",
    "light","load","lose","make","milk","move","no_interaction","open",
    "operate","pack","paint","park","pay","peel","pet","pick","pick_up",
    "point","pour","pull","push","race","read","release","repair","ride",
    "row","run","sail","scratch","serve","set","shear","sign","sip","sit_at",
    "sit_on","slide","smell","spin","squeeze","stab","stand_on","stand_under",
    "stick","stir","stop_at","straddle","swing","tag","talk_on","teach",
    "text_on","throw","tie","toast","train","turn","type_on","walk","wash",
    "watch","wave","wear","wield","zip","carry_on","pet_on","sit_under",
]
HICO_VERBS = HICO_VERBS[:NUM_VERBS]
if len(HICO_VERBS) < NUM_VERBS:
    HICO_VERBS += [f"verb_{i}" for i in range(len(HICO_VERBS), NUM_VERBS)]

# ════════════════════════════════════════════════════════════
# HOI TRANSFORMER DECODER
#   a) Object Queries: learnable embeddings for Human/Action/Object
#   b) Decoder layers attend to encoded features (memory)
#   c) Prediction heads -> human box, action class, object class+box
# ════════════════════════════════════════════════════════════
class HOITransformerDecoder(nn.Module):
    def __init__(self, dim=768, num_queries=16, num_verbs=117,
                 num_objects=80, num_layers=3, num_heads=8):
        super().__init__()
        # a) Object queries - one shared embedding per HOI slot
        self.queries = nn.Embedding(num_queries, dim)

        # b) Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim * 2, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # c) Prediction heads
        self.human_box_head   = nn.Linear(dim, 4)               # [cx,cy,w,h]
        self.action_head      = nn.Linear(dim, num_verbs)        # verb logits
        self.object_class_head = nn.Linear(dim, num_objects + 1) # +1 for "no object"
        self.object_box_head  = nn.Linear(dim, 4)               # [cx,cy,w,h]

    def forward(self, memory):
        B = memory.size(0)
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, dim]
        decoded = self.decoder(q, memory)                        # [B, Q, dim]

        return {
            "human_boxes"   : self.human_box_head(decoded).sigmoid(),     # [B,Q,4]
            "action_logits" : self.action_head(decoded),                  # [B,Q,117]
            "object_logits" : self.object_class_head(decoded),            # [B,Q,81]
            "object_boxes"  : self.object_box_head(decoded).sigmoid(),    # [B,Q,4]
        }


model = HOITransformerDecoder(
    dim=768, num_queries=NUM_QUERIES,
    num_verbs=NUM_VERBS, num_objects=NUM_OBJECTS).to(device)
model.eval()
print(f"\n✓ HOI Transformer Decoder initialised")
print(f"  Object queries  : {NUM_QUERIES}")
print(f"  Verb classes    : {NUM_VERBS}")
print(f"  Object classes  : {NUM_OBJECTS}")
print(f"  NOTE: untrained - predictions below are RANDOM, shapes are correct.")

# ════════════════════════════════════════════════════════════
# RUN DECODER
# ════════════════════════════════════════════════════════════
with torch.no_grad():
    outputs = model(refined_features)

print(f"\n===== DECODER OUTPUT SHAPES =====")
print(f"human_boxes    : {list(outputs['human_boxes'].shape)}")
print(f"action_logits  : {list(outputs['action_logits'].shape)}")
print(f"object_logits  : {list(outputs['object_logits'].shape)}")
print(f"object_boxes   : {list(outputs['object_boxes'].shape)}")

# ════════════════════════════════════════════════════════════
# DECODE TOP-1 PREDICTION (for display purposes - untrained, so random)
# ════════════════════════════════════════════════════════════
action_probs = torch.softmax(outputs["action_logits"], dim=-1)   # [1,Q,117]
object_probs = torch.softmax(outputs["object_logits"], dim=-1)   # [1,Q,81]

# Pick the query with highest combined confidence
combined_conf = action_probs.max(dim=-1).values * object_probs.max(dim=-1).values
best_query = combined_conf[0].argmax().item()

best_action_idx = action_probs[0, best_query].argmax().item()
best_object_idx = object_probs[0, best_query].argmax().item()

action_name = HICO_VERBS[best_action_idx]
object_name = COCO_CLASSES.get(best_object_idx, f"class_{best_object_idx}")

human_label = "Man" if len(human_detections) > 0 else "Person"

print(f"\n===== TOP-1 HOI PREDICTION (untrained - illustrative) =====")
print(f"  Human  -> {human_label}")
print(f"  Action -> {action_name}  (conf={action_probs[0,best_query,best_action_idx].item():.4f})")
print(f"  Object -> {object_name}  (conf={object_probs[0,best_query,best_object_idx].item():.4f})")
print(f"\n  Predicted triplet: ({human_label}, {action_name}, {object_name})")
print(f"\n  NOTE: This is the CORRECT OUTPUT FORMAT matching your")
print(f"  architecture diagram. Values are random/meaningless until")
print(f"  the model is trained on HICO-DET (Stage 9+).")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage8_output.pt")
torch.save({
    "outputs"          : {k: v.cpu() for k, v in outputs.items()},
    "predicted_triplet": (human_label, action_name, object_name),
    "human_detections" : human_detections,
    "object_detections": object_detections,
    "reasoning_output" : reasoning_output,
    "enhanced_image_np": enhanced_np,
    "decoder_state_dict": model.state_dict(),
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE - final summary image
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
ax.imshow(enhanced_np)
ax.axis('off')

title_text = (
    f"FINAL OUTPUT (Human, Action, Object)\n"
    f"( {human_label} , {action_name} , {object_name} )"
)
ax.set_title(title_text, fontsize=13, fontweight='bold', color='darkred')

info_text = (
    f"Detected humans  : {len(human_detections)}\n"
    f"Detected objects : {len(object_detections)}\n"
    f"Reasoning: {reasoning_output[:80]}...\n\n"
    f"[Pipeline stages 1-8 complete on this sample.\n"
    f"Decoder is untrained - this prediction is illustrative only.\n"
    f"Train the model (Stage 9+) for meaningful predictions.]"
)
fig.text(0.5, -0.05, info_text, ha='center', va='top', fontsize=9,
         bbox=dict(facecolor='lightyellow', alpha=0.8, pad=8))

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage8_final_output.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {viz_path}")

print("\n" + "=" * 60)
print("STAGE 8 COMPLETE - FULL 8-STAGE PIPELINE VERIFIED")
print("=" * 60)
print(f"Final triplet : ({human_label}, {action_name}, {object_name})")
print("\nAll 8 stages run correctly on a single image.")
print("Next step -> Build dataset loader (9_dataset.py) and")
print("             training loop (10_train.py) for the full")
print("             HICO-DET dataset (38,118 train / 9,658 test).")
print("=" * 60)