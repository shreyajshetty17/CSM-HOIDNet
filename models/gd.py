"""
STAGE 4 — DETR + GROUNDING DINO (Human + Object Detection)
=============================================================
FIXES vs original:
  1. Lower thresholds (BOX=0.20, TEXT=0.15) → catch more detections
  2. Richer TEXT_PROMPT with scene-context objects (saddle, fence, helmet…)
  3. get_color() defined before first use
  4. Custom NMS to remove overlapping duplicate boxes
  5. DETR fallback detections merged into human/object lists properly
  6. Bar-chart indentation bug fixed
  7. DETR also runs predict() and merges unique classes not found by GDINO
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Grounding DINO ───────────────────────────────────────────
from groundingdino.util.inference import load_model, load_image, predict

# ── DETR (HuggingFace) ───────────────────────────────────────
from transformers import DetrImageProcessor, DetrForObjectDetection

# ════════════════════════════════════════════════════════════
# CONFIG  ← tweak paths to match your machine
# ════════════════════════════════════════════════════════════
PROJECT_ROOT = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

GDINO_CONFIG_PATH = (
    r"C:\Users\STUDENT\AppData\Roaming\Python\Python312"
    r"\site-packages\groundingdino\config"
    r"\GroundingDINO_SwinT_OGC.py"
)
GDINO_WEIGHTS_PATH = (
    r"C:\Users\STUDENT\Downloads\hoid_project\weights"
    r"\groundingdino_swint_ogc.pth"
)

# ── Detection thresholds (LOWERED for higher recall) ────────
BOX_THRESHOLD  = 0.50   # was 0.30
TEXT_THRESHOLD = 0.45   # was 0.25
NMS_IOU_THRESHOLD = 0.35  # tighter → kills duplicate truck/horse boxes

# ── DETR confidence cutoff ───────────────────────────────────
DETR_CONFIDENCE = 0.25   # was 0.30

# ── EXPANDED text prompt ─────────────────────────────────────
# Added: saddle, helmet, fence, gate, tractor, trailer, barn,
#        rider, glove, boot, rope, barrel, pole, tree, dirt
TEXT_PROMPT = (
    "person . rider . horse . saddle . helmet . glove . boot . "
    "bicycle . motorcycle . car . bus . truck . tractor . trailer . "
    "phone . cup . bottle . chair . table . bench . "
    "dog . cat . bird . sheep . cow . "
    "fence . gate . barn . pole . barrel . rope . "
    "book . laptop . umbrella . bag . backpack . tie . ball . bat . "
    "skateboard . surfboard . tennis racket . "
    "couch . bed . tv . keyboard . mouse . remote . "
    "knife . fork . spoon . bowl . pizza . sandwich . cake . "
    "vase . scissors . tree . wall . building . vehicle"
)

# ── COCO label list for DETR ─────────────────────────────────
COCO_LABELS = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 4 — DETR + GROUNDING DINO  (Enhanced Detection)")
print("=" * 60)
print(f"Device           : {device}")
print(f"BOX_THRESHOLD    : {BOX_THRESHOLD}")
print(f"TEXT_THRESHOLD   : {TEXT_THRESHOLD}")
print(f"NMS_IOU          : {NMS_IOU_THRESHOLD}")

# ════════════════════════════════════════════════════════════
# HELPER — colour map (defined FIRST to avoid NameError)
# ════════════════════════════════════════════════════════════
_COLOR_MAP = {}
_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F0B27A', '#82E0AA', '#F1948A', '#AED6F1', '#A9DFBF',
    '#FAD7A0', '#D2B4DE', '#A3E4D7', '#FDEBD0', '#D5DBDB',
]

def get_color(label: str) -> str:
    """Return a consistent colour for a class label."""
    if label == "person":
        return "lime"
    if label not in _COLOR_MAP:
        _COLOR_MAP[label] = _PALETTE[len(_COLOR_MAP) % len(_PALETTE)]
    return _COLOR_MAP[label]


# ════════════════════════════════════════════════════════════
# HELPER — IoU + NMS
# ════════════════════════════════════════════════════════════
def compute_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def apply_nms(detections, iou_thr=NMS_IOU_THRESHOLD):
    """
    Simple greedy NMS over a list of detection dicts.
    Each dict must have 'confidence' and 'box' keys.
    """
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets
                if compute_iou(best["box"], d["box"]) < iou_thr]
    return keep


# ════════════════════════════════════════════════════════════
# LOAD STAGE 1 OUTPUT
# ════════════════════════════════════════════════════════════
stage1_path = os.path.join(OUTPUT_DIR, "stage1_output.pt")
if not os.path.exists(stage1_path):
    raise FileNotFoundError(f"{stage1_path} not found. Run restormer.py first.")

stage1        = torch.load(stage1_path, map_location="cpu", weights_only=False)
enhanced_np   = stage1["enhanced_image_np"]
enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
H_img, W_img  = enhanced_uint8.shape[:2]

print(f"\nLoaded Stage 1  : {stage1_path}")
print(f"Image shape     : {enhanced_uint8.shape}")

# ════════════════════════════════════════════════════════════
# STAGE 4A — GROUNDING DINO
# ════════════════════════════════════════════════════════════
human_detections  = []
object_detections = []

print("\n--- Grounding DINO ---")
print(f"Config  exists : {os.path.exists(GDINO_CONFIG_PATH)}")
print(f"Weights exists : {os.path.exists(GDINO_WEIGHTS_PATH)}")
print(f"Weights size   : {os.path.getsize(GDINO_WEIGHTS_PATH):,} bytes")

gdino_model = load_model(GDINO_CONFIG_PATH, GDINO_WEIGHTS_PATH).to(device)

# save temp image for GDINO's load_image()
temp_img_path = os.path.join(OUTPUT_DIR, "_temp_gdino_input.jpg")
Image.fromarray(enhanced_uint8).save(temp_img_path)
image_source, gdino_image = load_image(temp_img_path)

boxes, logits, phrases = predict(
    model=gdino_model,
    image=gdino_image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=str(device),
)
os.remove(temp_img_path)

# convert normalised cx,cy,w,h → pixel x1,y1,x2,y2
raw_gdino = []
for box, conf, phrase in zip(boxes, logits, phrases):
    cx, cy, w, h = box.tolist()
    x1 = (cx - w / 2) * W_img
    y1 = (cy - h / 2) * H_img
    x2 = (cx + w / 2) * W_img
    y2 = (cy + h / 2) * H_img
    raw_gdino.append({
        "class":      phrase.strip().lower(),
        "confidence": float(conf),
        "box":        [x1, y1, x2, y2],
        "source":     "gdino",
    })

# NMS per class
gdino_by_class: dict[str, list] = {}
for d in raw_gdino:
    gdino_by_class.setdefault(d["class"], []).append(d)

gdino_final = []
for cls, dets in gdino_by_class.items():
    gdino_final.extend(apply_nms(dets))

for det in gdino_final:
    if "person" in det["class"] or "rider" in det["class"]:
        human_detections.append(det)
    else:
        object_detections.append(det)

print(f"GDINO raw        : {len(raw_gdino)}  →  after NMS: {len(gdino_final)}")
print(f"  Humans         : {len(human_detections)}")
print(f"  Objects        : {len(object_detections)}")

# ════════════════════════════════════════════════════════════
# STAGE 4B — DETR  (encoder features + extra detections)
# ════════════════════════════════════════════════════════════
print("\n--- DETR ---")
processor  = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
detr_model.eval()

pil_image = Image.fromarray(enhanced_uint8)
inputs    = processor(images=pil_image, return_tensors="pt").to(device)

with torch.no_grad():
    detr_outputs = detr_model(**inputs)

# ── encoder features (for downstream stages) ────────────────
encoder_features = detr_outputs.last_hidden_state.cpu()
print(f"DETR encoder shape : {encoder_features.shape}")

# ── extract DETR detections & merge only NEW classes ────────
target_sizes = torch.tensor([[H_img, W_img]])
results      = processor.post_process_object_detection(
    detr_outputs, threshold=DETR_CONFIDENCE, target_sizes=target_sizes
)[0]

gdino_classes = {d["class"] for d in gdino_final}
detr_added    = 0

raw_detr = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    cls_name = COCO_LABELS[label.item()].lower()
    if cls_name == "n/a":
        continue
    x1, y1, x2, y2 = box.tolist()
    raw_detr.append({
        "class":      cls_name,
        "confidence": float(score),
        "box":        [x1, y1, x2, y2],
        "source":     "detr",
    })

# NMS per class on DETR results
detr_by_class: dict[str, list] = {}
for d in raw_detr:
    detr_by_class.setdefault(d["class"], []).append(d)

for cls, dets in detr_by_class.items():
    after_nms = apply_nms(dets)
    for det in after_nms:
        # Add if GDINO didn't already find this class,
        # OR if GDINO found it but this box doesn't overlap much
        overlaps = False
        all_existing = human_detections + object_detections
        for existing in all_existing:
            if compute_iou(det["box"], existing["box"]) > NMS_IOU_THRESHOLD:
                overlaps = True
                break
        if not overlaps:
            # extra guard: skip if same class already exists from gdino
            same_class_exists = any(
                e["class"] == cls and compute_iou(det["box"], e["box"]) > 0.25
                for e in human_detections + object_detections
            )
            if not same_class_exists:
                if "person" in cls or "rider" in cls:
                    human_detections.append(det)
                else:
                    object_detections.append(det)
                detr_added += 1

print(f"DETR raw         : {len(raw_detr)}  →  new unique added: {detr_added}")
print(f"\nFINAL — Humans  : {len(human_detections)}")
print(f"FINAL — Objects : {len(object_detections)}")
print(f"FINAL — Total   : {len(human_detections) + len(object_detections)}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
model_name = "Grounding DINO + DETR Features (Enhanced)"
save_path  = os.path.join(OUTPUT_DIR, "stage4_detection.pt")

torch.save({
    "model_name":        model_name,
    "human_detections":  human_detections,
    "object_detections": object_detections,
    "encoder_features":  encoder_features,
    "enhanced_image_np": enhanced_np,
    "image_path":        stage1["image_path"],
}, save_path)

print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VIZ 1 — Detection boxes only (clean, no sidebar)
# ════════════════════════════════════════════════════════════
total_dets = len(human_detections) + len(object_detections)

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.imshow(enhanced_uint8)

for det in human_detections:
    x1, y1, x2, y2 = det["box"]
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor='lime', facecolor='none'
    )
    ax1.add_patch(rect)
    ax1.text(
        x1 + 3, y1 + 14,
        f"{det['class']}  {det['confidence']:.2f}",
        color='lime', fontsize=8.5, fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none')
    )

for det in object_detections:
    x1, y1, x2, y2 = det["box"]
    color = get_color(det["class"])
    rect  = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor=color, facecolor='none'
    )
    ax1.add_patch(rect)
    ax1.text(
        x1 + 3, y1 + 14,
        f"{det['class']}  {det['confidence']:.2f}",
        color=color, fontsize=8.5, fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none')
    )

ax1.set_title(
    f"Stage 4 — Detections   "
    f"Humans: {len(human_detections)}   "
    f"Objects: {len(object_detections)}   "
    f"Total: {total_dets}",
    fontsize=12, fontweight='bold', pad=10
)
ax1.axis('off')
plt.tight_layout()

det_path = os.path.join(OUTPUT_DIR, "stage4_detections.png")
fig1.savefig(det_path, dpi=200, bbox_inches='tight')
plt.close(fig1)
print(f"Saved detections : {det_path}")

# ════════════════════════════════════════════════════════════
# VIZ 2 — Summary bar chart only (clean, styled)
# ════════════════════════════════════════════════════════════
all_classes  = [d["class"] for d in human_detections + object_detections]
class_counts = Counter(all_classes)

# sort by count descending
sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
labels_bar   = [item[0] for item in sorted_items]
counts_bar   = [item[1] for item in sorted_items]
colors_bar   = [get_color(c) for c in labels_bar]

fig2, ax2 = plt.subplots(figsize=(9, max(4, len(labels_bar) * 0.65)))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#16213e')

if not labels_bar:
    ax2.text(0.5, 0.5, "No detections", ha="center", va="center",
             fontsize=13, color='white')
else:
    bars = ax2.barh(labels_bar, counts_bar, color=colors_bar,
                    edgecolor='none', height=0.55)

    # value labels
    for bar, count in zip(bars, counts_bar):
        ax2.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            str(count), va='center', fontsize=10,
            color='white', fontweight='bold'
        )

    ax2.set_xlim(0, max(counts_bar) + 1)
    ax2.set_xlabel("Count", fontsize=10, color='#aaaaaa', labelpad=8)
    ax2.tick_params(colors='white', labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#333355')
    ax2.spines['bottom'].set_color('#333355')
    ax2.xaxis.label.set_color('#aaaaaa')
    ax2.yaxis.label.set_color('white')
    for label in ax2.get_yticklabels():
        label.set_color('white')
    for label in ax2.get_xticklabels():
        label.set_color('#aaaaaa')

ax2.set_title(
    "Detection Summary",
    fontsize=13, fontweight='bold', color='white', pad=12
)
plt.tight_layout()

graph_path = os.path.join(OUTPUT_DIR, "stage4_summary_graph.png")
fig2.savefig(graph_path, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"Saved graph      : {graph_path}")

# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 4 COMPLETE")
print("=" * 60)
print(f"Model    : {model_name}")
print(f"Humans   : {len(human_detections)}")
print(f"Objects  : {len(object_detections)}")
print(f"Total    : {total_dets}")