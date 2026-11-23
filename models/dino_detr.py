"""
STAGE 4 — DINO-DETR (Human + Object Detection)
================================================
Architecture role : Enhanced Image -> Detected Human + Object boxes
Input  : enhanced image (outputs/stage1_output.pt)
Output : human/object detections + encoder features
         saved to outputs/stage4_output.pt

Uses REAL DINO if available, else DETR fallback.
Threshold lowered to 0.3 for more complete detection.
All detected objects shown with colored boxes.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
PROJECT_ROOT    = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "outputs")
DINO_ROOT       = os.path.join(PROJECT_ROOT, "DINO")
DINO_CONFIG     = os.path.join(DINO_ROOT, "config", "DINO", "DINO_4scale.py")
DINO_CHECKPOINT = os.path.join(DINO_ROOT, "checkpoints", "checkpoint0033_4scale.pth")

# ← LOWER THRESHOLD = MORE DETECTIONS
CONFIDENCE_THRESHOLD = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 4 — DINO-DETR (Human + Object Detection)")
print("=" * 60)
print(f"Device              : {device}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

# ════════════════════════════════════════════════════════════
# LOAD STAGE 1 OUTPUT
# ════════════════════════════════════════════════════════════
stage1_path = os.path.join(OUTPUT_DIR, "stage1_output.pt")
if not os.path.exists(stage1_path):
    raise FileNotFoundError(f"{stage1_path} not found. Run 1_restormer.py first.")

stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
enhanced_np    = stage1["enhanced_image_np"]
enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)

print(f"\nLoaded : {stage1_path}")
print(f"Image  : {enhanced_uint8.shape}")

# ════════════════════════════════════════════════════════════
# COCO CLASS NAMES
# ════════════════════════════════════════════════════════════
COCO_NAMES = [
    'N/A','person','bicycle','car','motorcycle','airplane','bus','train',
    'truck','boat','traffic light','fire hydrant','N/A','stop sign',
    'parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','N/A','backpack','umbrella','N/A',
    'N/A','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','N/A','wine glass','cup','fork',
    'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli',
    'carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','N/A','dining table','N/A','N/A','toilet','N/A','tv','laptop',
    'mouse','remote','keyboard','cell phone','microwave','oven','toaster',
    'sink','refrigerator','N/A','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

# Color map: person=lime, animals=red, vehicles=cyan, others=mediumpurple
ANIMALS   = {"horse","dog","cat","cow","sheep","elephant","bear","zebra","giraffe","bird"}
VEHICLES  = {"bicycle","car","motorcycle","airplane","bus","train","truck","boat"}

def get_color(class_name):
    if class_name == "person":   return "lime"
    if class_name in ANIMALS:    return "red"
    if class_name in VEHICLES:   return "cyan"
    return "mediumpurple"

# ════════════════════════════════════════════════════════════
# CHECK DINO AVAILABILITY
# ════════════════════════════════════════════════════════════
DINO_AVAILABLE = (
    os.path.exists(DINO_ROOT)
    and os.path.exists(DINO_CONFIG)
    and os.path.exists(DINO_CHECKPOINT)
)

if DINO_AVAILABLE:
    try:
        sys.path.insert(0, DINO_ROOT)
        from main import build_model_main
        from util.slconfig import SLConfig
        from util import box_ops
        print("\n✓ Real DINO-4scale loading ...")
    except Exception as e:
        print(f"\n! DINO import failed ({e}). Falling back to DETR.")
        DINO_AVAILABLE = False
else:
    print(f"\n! DINO checkpoint not found. Using DETR fallback.")

from PIL import Image

# ════════════════════════════════════════════════════════════
# OPTION A — REAL DINO-DETR
# ════════════════════════════════════════════════════════════
if DINO_AVAILABLE:
    args = SLConfig.fromfile(DINO_CONFIG)
    args.device = str(device)
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(DINO_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print("✓ Real DINO-4scale loaded")

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    pil_image  = Image.fromarray(enhanced_uint8)
    W, H       = pil_image.size
    img_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    pred_logits = outputs["pred_logits"]
    pred_boxes  = outputs["pred_boxes"]
    probs       = pred_logits.sigmoid()
    scores, labels = probs.max(-1)

    keep        = scores[0] > CONFIDENCE_THRESHOLD
    kept_boxes  = pred_boxes[0][keep]
    kept_scores = scores[0][keep]
    kept_labels = labels[0][keep]

    boxes_xyxy  = box_ops.box_cxcywh_to_xyxy(kept_boxes)
    boxes_xyxy  = boxes_xyxy * torch.tensor([W, H, W, H], device=device)

    encoder_features = outputs.get("encoder_output", pred_logits)
    model_name = "DINO-4scale (real)"

# ════════════════════════════════════════════════════════════
# OPTION B — DETR FALLBACK
# ════════════════════════════════════════════════════════════
else:
    from transformers import DetrImageProcessor, DetrForObjectDetection

    print("\nLoading DETR (facebook/detr-resnet-50) ...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model     = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model     = model.to(device)
    model.eval()
    print("✓ DETR loaded")

    pil_image = Image.fromarray(enhanced_uint8)
    W, H      = pil_image.size
    inputs    = processor(images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[H, W]]).to(device)
    results      = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes,
        threshold=CONFIDENCE_THRESHOLD)[0]

    kept_scores = results["scores"]
    kept_labels = results["labels"]
    boxes_xyxy  = results["boxes"]

    encoder_features = outputs.last_hidden_state \
        if hasattr(outputs, "last_hidden_state") else None
    model_name = "DETR-ResNet50"

# ════════════════════════════════════════════════════════════
# PARSE DETECTIONS — all classes kept, separated by type
# ════════════════════════════════════════════════════════════
human_detections  = []
object_detections = []

for score, label, box in zip(kept_scores, kept_labels, boxes_xyxy):
    label_idx  = label.item()
    class_name = COCO_NAMES[label_idx] \
        if label_idx < len(COCO_NAMES) else f"class_{label_idx}"
    if class_name == "N/A":
        continue
    box_list = box.detach().cpu().numpy().tolist()
    conf = score.item()
    det  = {"class": class_name, "confidence": conf, "box": box_list}
    if class_name == "person":
        human_detections.append(det)
    else:
        object_detections.append(det)

# ════════════════════════════════════════════════════════════
# PRINT RESULTS
# ════════════════════════════════════════════════════════════
print(f"\n===== DETECTION RESULTS ({model_name}) =====")
print(f"Humans   : {len(human_detections)}")
for h in human_detections:
    print(f"  person  conf={h['confidence']:.3f}  box={[round(b,1) for b in h['box']]}")

print(f"\nObjects  : {len(object_detections)}")
for o in object_detections:
    print(f"  {o['class']:15s} conf={o['confidence']:.3f}  box={[round(b,1) for b in o['box']]}")

print(f"\nTotal detections : {len(human_detections) + len(object_detections)}")
print(f"Encoder features : "
      f"{list(encoder_features.shape) if encoder_features is not None else 'N/A'}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage4_output.pt")
torch.save({
    "model_name"        : model_name,
    "human_detections"  : human_detections,
    "object_detections" : object_detections,
    "encoder_features"  : encoder_features.cpu() if encoder_features is not None else None,
    "enhanced_image_np" : enhanced_np,
    "image_path"        : stage1["image_path"],
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE — every detection with correct color + label
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# LEFT — original enhanced image with all boxes
axes[0].imshow(enhanced_uint8)

for det in human_detections:
    x1, y1, x2, y2 = det["box"]
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=2.5, edgecolor='lime', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(x1, max(y1-5, 0), f"person {det['confidence']:.2f}",
                 color='lime', fontsize=8, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

for det in object_detections:
    x1, y1, x2, y2 = det["box"]
    color = get_color(det["class"])
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=2.5, edgecolor=color, facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(x1, max(y1-5, 0), f"{det['class']} {det['confidence']:.2f}",
                 color=color, fontsize=8, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

axes[0].set_title(
    f"Stage 4: {model_name}\n"
    f"{len(human_detections)} person(s)   {len(object_detections)} object(s)   "
    f"threshold={CONFIDENCE_THRESHOLD}",
    fontsize=11, fontweight='bold')
axes[0].axis('off')

# RIGHT — detection summary bar chart
all_classes = [d["class"] for d in human_detections + object_detections]
from collections import Counter
class_counts = Counter(all_classes)
labels_bar   = list(class_counts.keys())
counts_bar   = list(class_counts.values())
colors_bar   = [get_color(c) for c in labels_bar]

bars = axes[1].barh(labels_bar, counts_bar, color=colors_bar, edgecolor='white')
axes[1].set_xlabel("Count", fontsize=10)
axes[1].set_title("Detection Summary\n(all detected classes)", fontsize=11, fontweight='bold')
axes[1].set_facecolor('#1a1a1a')
axes[1].tick_params(colors='white')
axes[1].xaxis.label.set_color('white')
axes[1].title.set_color('white')
for bar, count in zip(bars, counts_bar):
    axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', color='white', fontsize=9)

fig.patch.set_facecolor('#111111')
plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage4_visualization.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {viz_path}")

# ════════════════════════════════════════════════════════════
# LEGEND EXPLANATION
# ════════════════════════════════════════════════════════════
print("\n===== BOX COLOR LEGEND =====")
print("  Lime          = Person / Human")
print("  Red           = Animal (horse, dog, cat, etc.)")
print("  Cyan          = Vehicle (car, bicycle, etc.)")
print("  Mediumpurple  = Other object (tie, bag, etc.)")
print("\nNOTE: Horse IS correctly an 'object' in HOI context.")
print("  In HICO-DET, humans interact WITH horses (ride, feed, etc.)")
print("  So horse = object is the correct classification for HOI.")

print("\n" + "=" * 60)
print("STAGE 4 COMPLETE")
print("=" * 60)
print(f"Model    : {model_name}")
print(f"Humans   : {len(human_detections)}")
print(f"Objects  : {len(object_detections)}")
print(f"Total    : {len(human_detections) + len(object_detections)}")
print("Next     -> Stage 5: SMAB (Subtle Motion Attention Block)")
print("=" * 60)