# ============================================================

# STAGE 4 — GROUNDING DINO

# ============================================================

from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import AutoModelForZeroShotObjectDetection

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


print("\nLoading Grounding DINO...")

MODEL_ID = "IDEA-Research/grounding-dino-base"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

model = model.to(device)
model.eval()

print("✓ Grounding DINO loaded")

pil_image = Image.fromarray(enhanced_unit8)
W, H = pil_image.size

# Open-vocabulary prompts for HOI

TEXT_PROMPT = (
"person . bicycle . motorcycle . horse . dog . cat . backpack . "
"handbag . suitcase . skateboard . surfboard . tennis racket . "
"sports ball . bottle . cup . wine glass . chair . couch . "
"laptop . keyboard . mouse . cell phone . book ."
)

inputs = processor(
images=pil_image,
text=TEXT_PROMPT,
return_tensors="pt"
).to(device)

with torch.no_grad():
 outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
outputs,
inputs.input_ids,
box_threshold=0.30,
text_threshold=0.25,
target_sizes=[(H, W)]
)[0]

boxes_xyxy = results["boxes"]
kept_scores = results["scores"]
labels_text = results["labels"]

human_detections = []
object_detections = []

for score, label, box in zip(
kept_scores,
labels_text,
boxes_xyxy):


 class_name = str(label).strip().lower()

det = {
    "class": class_name,
    "confidence": float(score),
    "box": box.cpu().numpy().tolist()
}

if class_name == "person":
    human_detections.append(det)
else:
    object_detections.append(det)


encoder_features = None
model_name = "Grounding DINO"
