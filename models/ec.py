"""
STAGE 6 — EACRM (Environment-Aware Commonsense Reasoning)
===========================================================
Matches architecture diagram EXACTLY:

a) CLIP Image Encoder   -> encode global image semantics
b) Scene Transformer    -> model scene layout, objects, environment
c) Knowledge Prompt     -> Q: Why is the person doing X in this scene?
                           Knowledge Base (social behavior, context)
d) Reasoning Output     -> "The person is likely doing X to achieve Y."

Input  : enhanced image + detections from Stage 4
Output : scene embedding [1, 256] + Q&A style reasoning text
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

from ollama import chat

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
PROJECT_ROOT = r"C:\Users\STUDENT\Downloads\hoid_project"
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 6 — EACRM (Environment-Aware Commonsense Reasoning)")
print("=" * 60)
print(f"Device : {device}")

# ════════════════════════════════════════════════════════════
# LOAD INPUTS
# ════════════════════════════════════════════════════════════
stage4 = torch.load(os.path.join(OUTPUT_DIR, "stage4_output.pt"),
                    map_location="cpu", weights_only=False)
enhanced_np       = stage4["enhanced_image_np"]
enhanced_uint8    = (enhanced_np * 255).astype(np.uint8)
human_detections  = stage4["human_detections"]
object_detections = stage4["object_detections"]

print(f"\nLoaded Stage 4")
print(f"Humans  : {len(human_detections)}")
print(f"Objects : {[d['class'] for d in object_detections]}")

# ════════════════════════════════════════════════════════════
# a) CLIP IMAGE ENCODER
#    Encode global image semantics (what is actually in the image)
# ════════════════════════════════════════════════════════════
import open_clip
from PIL import Image

print("\n[a] CLIP Image Encoder - encoding global image semantics ...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

tokenizer   = open_clip.get_tokenizer('ViT-B-32')
pil_image   = Image.fromarray(enhanced_uint8)
image_input = preprocess(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
    image_feature = clip_model.encode_image(image_input)
    image_feature = F.normalize(image_feature.float(), dim=-1)

print(f"    CLIP feature shape : {list(image_feature.shape)}")

# ════════════════════════════════════════════════════════════
# b) SCENE TRANSFORMER
#    Model scene layout, objects, environment
# ════════════════════════════════════════════════════════════
print("\n[b] Scene Transformer - modelling scene layout ...")

class SceneTransformer(nn.Module):
    def __init__(self, clip_dim=512, scene_dim=256, num_layers=3, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(clip_dim, scene_dim)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=scene_dim, nhead=num_heads,
            dim_feedforward=scene_dim * 4, batch_first=True)
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj   = nn.Linear(scene_dim, scene_dim)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x)
        return self.out_proj(x.squeeze(1))

scene_transformer = SceneTransformer().to(device)
scene_transformer.eval()

with torch.no_grad():
    scene_embedding = scene_transformer(image_feature)

print(f"    Scene embedding : {list(scene_embedding.shape)}")

# ════════════════════════════════════════════════════════════
# c) KNOWLEDGE PROMPT (LLM Style)
#    Step 1: Use CLIP to find WHAT the person is doing (visual grounding)
#    Step 2: Build "Q: Why is the person doing X?" prompt
#    Step 3: Answer using Knowledge Base (context rules)
# ════════════════════════════════════════════════════════════
print("\n[c] Knowledge Prompt (LLM Style) ...")

# HOI candidate descriptions — CLIP finds the best visual match
HOI_CANDIDATES = [
    "a person cleaning a bus",
    "a person washing a bus",
    "a person maintaining a bus",
    "a person spraying water on a bus",
    "a person boarding a bus",
    "a person riding a bus",
    "a person exiting a bus",
    "a person standing near a bus",
    "a person driving a bus",
    "a person washing a car",
    "a person cleaning a car",
    "a person riding a bicycle",
    "a person riding a motorcycle",
    "a person riding a horse",
    "a person using a cell phone",
    "a person texting on a phone",
    "a person holding a phone",
    "a person drinking from a cup",
    "a person eating food",
    "a person carrying a bag",
    "a person reading a book",
    "a person using a laptop",
    "a person holding an umbrella",
    "a person holding a hose",
    "a person holding a mop",
    "a person playing sports",
    "a person kicking a ball",
    "a person walking a dog",
    "a person petting a dog",
    "a person sitting on a chair",
    "a person sitting at a dining table",
    "a person working outdoors",
    "a person doing maintenance work",
]

# CLIP finds what is actually happening in the image
with torch.no_grad():
    text_tokens   = tokenizer(HOI_CANDIDATES).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features.float(), dim=-1)

similarities = (image_feature @ text_features.T)[0]
probs        = F.softmax(similarities * 100, dim=0)
top5_vals, top5_idxs = torch.topk(probs, k=5)
top5_labels  = [HOI_CANDIDATES[i] for i in top5_idxs.tolist()]
top5_probs   = top5_vals.tolist()

# Best visual match = what CLIP sees in the image
best_match  = top5_labels[0]
# Extract action from best match e.g. "a person cleaning a bus" -> "cleaning a bus"
action_phrase = best_match.replace("a person ", "")   # e.g. "cleaning a bus"
action_verb   = action_phrase.split()[0]               # e.g. "cleaning"
action_object = " ".join(action_phrase.split()[1:])    # e.g. "a bus"

print(f"    Top-5 visual matches:")
for i, (label, prob) in enumerate(zip(top5_labels, top5_probs)):
    print(f"      {i+1}. {label:40s} score={prob:.4f}")

print(f"\n    Best match : '{best_match}'")

# ==========================================================
# PHI-3 REASONING
# ==========================================================

detected_objects = [d['class'] for d in object_detections]

prompt = f"""
You are an expert Human Object Interaction reasoning model.

Humans detected:
{len(human_detections)}

Objects detected:
{detected_objects}

Visual action from CLIP:
{action_phrase}

Question:
Why is the person performing this action?

Answer using:

Environment:
Social Context:
Reason:
Reasoning Output:
"""

print("\nSending prompt to Phi-3...\n")

response = chat(
    model="phi3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

llm_answer = response["message"]["content"]

knowledge_prompt_Q = (
    f"Q: Why is the person {action_phrase}?"
)

knowledge_prompt_A = llm_answer

print("\nPhi-3 Output:\n")
print(llm_answer)
# ════════════════════════════════════════════════════════════
# d) REASONING OUTPUT
#    Final commonsense explanation — matches diagram's output box
# ════════════════════════════════════════════════════════════
num_people   = len(human_detections)
people_str   = f"{num_people} person(s)" if num_people > 0 else "a person"

reasoning_output = llm_answer

full_reasoning = llm_answer

print(f"\n[d] Reasoning Output:")
print(f"    {'─'*50}")
print(f"    {knowledge_prompt_Q}")
print(f"    {knowledge_prompt_A}")
print(f"    {'─'*50}")
print(f"    Full: {full_reasoning}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage6_output.pt")
torch.save({
    "global_scene_feature"        : image_feature.cpu(),
    "scene_embedding"             : scene_embedding.cpu(),
    "best_match"                  : best_match,
    "action_phrase"               : action_phrase,
    "knowledge_prompt_Q"          : knowledge_prompt_Q,
    "knowledge_prompt_A"          : knowledge_prompt_A,
    "reasoning_output"            : reasoning_output,
    "full_reasoning"              : full_reasoning,
    "top5_labels"                 : top5_labels,
    "top5_probs"                  : top5_probs,
    "human_detections"            : human_detections,
    "object_detections"           : object_detections,
    "enhanced_image_np"           : enhanced_np,
    "image_path"                  : stage4["image_path"],
    "scene_transformer_state_dict": scene_transformer.state_dict(),
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE — exact match to architecture diagram layout
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 7))
fig.patch.set_facecolor('#f0f4f8')

# ── a) CLIP Image (left) ────────────────────────────────────
ax1 = fig.add_subplot(1, 4, 1)
ax1.imshow(enhanced_uint8)
ax1.set_title("a) CLIP Image Encoder\nGlobal Image Semantics",
               fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#dce8ff', alpha=0.8))
ax1.axis('off')

# ── b) Scene Transformer (bar chart of scene embedding norms) ──
ax2 = fig.add_subplot(1, 4, 2)
scene_np = scene_embedding[0].detach().cpu().numpy()
chunks   = np.array_split(scene_np, 8)
chunk_norms = [np.linalg.norm(c) for c in chunks]
ax2.bar(range(len(chunk_norms)), chunk_norms, color='#6c8ebf')
ax2.set_title("b) Scene Transformer\nScene Layout Encoding",
               fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#dce8ff', alpha=0.8))
ax2.set_xlabel("Feature chunks", fontsize=8)
ax2.set_ylabel("Norm", fontsize=8)

# ── c) Knowledge Prompt (Q&A box like diagram) ─────────────
ax3 = fig.add_subplot(1, 4, 3)
ax3.axis('off')
ax3.set_facecolor('#fff9e6')

# Top candidate bar chart inside
top5_short = [l.replace("a person ", "")[:25] for l in top5_labels]
bars = ax3.barh(range(4, -1, -1), top5_probs,
                color=['#27ae60'] + ['#3498db'] * 4,
                transform=ax3.transData)

prompt_text = knowledge_prompt_Q + "\n\n" + knowledge_prompt_A[:300]

ax3.text(0.05, 0.95, prompt_text,
         transform=ax3.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9, pad=6))

ax3.set_title("c) Knowledge Prompt (LLM Style)\nVisual matching + Knowledge Base",
               fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8))

# ── d) Reasoning Output (green box like diagram) ────────────
ax4 = fig.add_subplot(1, 4, 4)
ax4.axis('off')
ax4.set_facecolor('#e8f8e8')

output_text = (
    f"d) Reasoning Output\n\n"
    f"{reasoning_output[:500]}"
)
ax4.text(0.5, 0.5, output_text,
         transform=ax4.transAxes,
         fontsize=10, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.9, pad=10),
         color='#155724', fontweight='bold')

ax4.set_title("d) Reasoning Output\n(Commonsense Explanation)",
               fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))

plt.suptitle(
    "Stage 6 — EACRM (Environment-Aware Commonsense Reasoning)\n"
    "CLIP + Scene Transformer + Knowledge Prompt (LLM Style)",
    fontsize=12, fontweight='bold', y=1.02)

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "stage6_visualization.png")
plt.savefig(viz_path, dpi=130, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {viz_path}")

print("\n" + "=" * 60)
print("STAGE 6 COMPLETE")
print("=" * 60)
print(f"Scene embedding  : {list(scene_embedding.shape)}")
print(f"Best HOI match   : '{best_match}'")
print(f"Knowledge Prompt : {knowledge_prompt_Q}")
print(f"Reasoning Output : {reasoning_output}")
print("Next   -> Stage 7: Semi-Supervised Contrastive Learning")
print("=" * 60)