# models/eacrm.py

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
from ollama import chat
from PIL import Image

warnings.filterwarnings("ignore")

# ==========================================================

# CONFIG

# ==========================================================

PROJECT_ROOT = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("STAGE 6 — EACRM (Phi-3 Powered)")
print("=" * 60)
print("Device:", device)

# ==========================================================

# LOAD STAGE 4 OUTPUT

# ==========================================================

stage4_path = os.path.join(
OUTPUT_DIR,
"stage4_output.pt"
)

stage4 = torch.load(
stage4_path,
weights_only=False
)

enhanced_np = stage4["enhanced_image_np"]
human_detections = stage4["human_detections"]
object_detections = stage4["object_detections"]

print("\nLoaded Stage 4")
print("Humans :", len(human_detections))
print("Objects:", len(object_detections))

# ==========================================================

# IMAGE PREP

# ==========================================================

enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
pil_image = Image.fromarray(enhanced_uint8)

# ==========================================================

# CLIP FEATURE EXTRACTION

# ==========================================================

print("\nLoading CLIP...")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
"ViT-B-32",
pretrained="openai"
)

clip_model = clip_model.to(device)
clip_model.eval()

for p in clip_model.parameters():
 p.requires_grad = False

image_input = preprocess(
pil_image
).unsqueeze(0).to(device)

with torch.no_grad():
 global_scene_feature = clip_model.encode_image(
image_input
)

global_scene_feature = (
global_scene_feature
/ global_scene_feature.norm(dim=-1, keepdim=True)
)

print(
"CLIP feature shape:",
list(global_scene_feature.shape)
)

class SceneTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(
            512,
            256
        )

        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=3
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x.squeeze(1)
    # ==========================================================
# SCENE TRANSFORMER
# ==========================================================

scene_transformer = SceneTransformer().to(device)
scene_transformer.eval()

with torch.no_grad():
    scene_embedding = scene_transformer(
        global_scene_feature.float()
    )

print(
    "Scene embedding shape:",
    list(scene_embedding.shape)
)
# ==========================================================

# PREPARE KNOWLEDGE PROMPT

# ==========================================================

object_names = []

for obj in object_detections:
     if "class" in obj:
      object_names.append(obj["class"])

if len(object_names) == 0:
     object_names.append("unknown")

scene_prompt = f"""
You are an expert Human Object Interaction analyst.

Detected humans:
{len(human_detections)}

Detected objects:
{", ".join(object_names)}

Analyze the scene and provide:

1. Scene type
2. Human-object interactions
3. Likely activity
4. Commonsense explanation

Keep the answer under 150 words.
"""

# ==========================================================

# PHI-3 REASONING

# ==========================================================

print("\nGenerating reasoning using Phi-3...")

response = chat(
model="phi3",
messages=[
{
"role": "user",
"content": scene_prompt
}
]
)

reasoning_output = response["message"]["content"]

print("\nReasoning Output:")
print("-" * 60)
print(reasoning_output)
print("-" * 60)

# ==========================================================

# SAVE OUTPUT

# ==========================================================

save_path = os.path.join(
OUTPUT_DIR,
"stage6_output.pt"
)

torch.save(
{
"global_scene_feature":
global_scene_feature.cpu(),


    "scene_embedding":
        scene_embedding.cpu(),

    "reasoning_output":
        reasoning_output,

    "human_detections":
        human_detections,

    "object_detections":
        object_detections,

    "enhanced_image_np":
        enhanced_np,
},
save_path


)

print("\nSaved:", save_path)

# ==========================================================

# VISUALIZATION

# ==========================================================

fig, ax = plt.subplots(
figsize=(10, 8)
)

ax.imshow(enhanced_uint8)
ax.axis("off")

fig.text(
0.5,
0.02,
reasoning_output,
ha="center",
fontsize=10,
wrap=True,
bbox=dict(
facecolor="lightyellow",
alpha=0.9
)
)

viz_path = os.path.join(
OUTPUT_DIR,
"stage6_visualization.png"
)

plt.savefig(
viz_path,
dpi=150,
bbox_inches="tight"
)

plt.close()

print("Saved:", viz_path)

print("\nStage 6 COMPLETE")
print("Reasoning generated by Phi-3")
