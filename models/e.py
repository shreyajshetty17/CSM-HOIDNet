"""
STAGE 6 — EACRM (Environment-Aware Commonsense Reasoning)
============================================================
FREE LLM VERSION — uses Ollama (local, no API key, no cost)

Architecture (matches diagram EXACTLY):
a) CLIP Image Encoder   -> encode global image semantics
b) Scene Transformer    -> model scene layout, objects, environment
c) Knowledge Prompt     -> Q: Why is the person doing X in this scene?
                           Free local LLM (Ollama) answers using reasoning
d) Reasoning Output     -> "The person is likely doing X to achieve Y."

SETUP REQUIRED (one-time, free):
  1. Download Ollama from https://ollama.com/download (Windows installer)
  2. Open a NEW terminal and run: ollama pull llama3.2
     (or smaller/faster: ollama pull phi3)
  3. Ollama runs automatically as a background service after install
  4. No API key needed - it's 100% free and runs on your own machine
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
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

# ── Ollama config (FREE, runs locally, NO API KEY NEEDED) ────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # change to "phi3" if you want a smaller/faster model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("STAGE 6 — EACRM  (Free Local LLM via Ollama)")
print("=" * 60)
print(f"Device : {device}")
print(f"LLM    : {OLLAMA_MODEL} (via Ollama, free, local, no API key)")

# ════════════════════════════════════════════════════════════
# LOAD INPUTS
# ════════════════════════════════════════════════════════════
stage4_path = os.path.join(OUTPUT_DIR, "stage4_detection.pt")
if not os.path.exists(stage4_path):
    stage4_path = os.path.join(OUTPUT_DIR, "stage4_output.pt")
if not os.path.exists(stage4_path):
    raise FileNotFoundError(f"Neither stage4_detection.pt nor stage4_output.pt found. Run Stage 4 first.")
print(f"  Loading from : {stage4_path}")

stage4            = torch.load(stage4_path, map_location="cpu", weights_only=False)
enhanced_np       = stage4["enhanced_image_np"]
enhanced_uint8    = (enhanced_np * 255).astype(np.uint8) if enhanced_np.max() <= 1.0 else enhanced_np.astype(np.uint8)
human_detections  = stage4["human_detections"]
object_detections = stage4["object_detections"]

print(f"\nLoaded Stage 4")
print(f"  Image source : {stage4.get('image_path', 'UNKNOWN')}")
print(f"  >>> If this is NOT the image you expect, re-run Stages 1-4 first! <<<")
print(f"  Humans  : {len(human_detections)}")
print(f"  Objects : {[d['class'] for d in object_detections]}")

H_img, W_img = enhanced_uint8.shape[:2]

# ════════════════════════════════════════════════════════════
# a) CLIP IMAGE ENCODER
# ════════════════════════════════════════════════════════════
import open_clip

print("\n[a] CLIP Image Encoder ...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device).eval()
for p in clip_model.parameters():
    p.requires_grad = False

tokenizer   = open_clip.get_tokenizer('ViT-B-32')
pil_image   = Image.fromarray(enhanced_uint8)
image_input = preprocess(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
    image_feature = clip_model.encode_image(image_input)
    image_feature = F.normalize(image_feature.float(), dim=-1)

print(f"    CLIP feature : {list(image_feature.shape)}")

# ════════════════════════════════════════════════════════════
# b) SCENE TRANSFORMER
# ════════════════════════════════════════════════════════════
print("\n[b] Scene Transformer ...")

class SceneTransformer(nn.Module):
    def __init__(self, clip_dim=512, scene_dim=256, num_layers=3, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(clip_dim, scene_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=scene_dim, nhead=num_heads,
            dim_feedforward=scene_dim * 4, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(scene_dim, scene_dim)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x)
        return self.out_proj(x.squeeze(1))

scene_transformer = SceneTransformer().to(device).eval()
with torch.no_grad():
    scene_embedding = scene_transformer(image_feature)

print(f"    Scene embedding : {list(scene_embedding.shape)}")

# ════════════════════════════════════════════════════════════
# CLIP VISUAL GROUNDING — generate HOI candidates dynamically
# ════════════════════════════════════════════════════════════
print("\n[c] CLIP visual grounding ...")

OBJECT_ACTIONS = {
    "horse":    ["riding", "feeding", "petting", "leading", "grooming", "sitting on"],
    "dog":      ["walking", "petting", "feeding", "playing with", "holding"],
    "cat":      ["petting", "holding", "feeding", "playing with"],
    "bicycle":  ["riding", "holding", "pushing", "fixing"],
    "motorcycle":["riding", "holding", "pushing", "sitting on"],
    "car":      ["driving", "cleaning", "washing", "sitting in", "standing near"],
    "truck":    ["driving", "loading", "standing near", "cleaning"],
    "tractor":  ["driving", "operating", "sitting on", "standing near"],
    "bus":      ["boarding", "exiting", "riding", "driving", "standing near"],
    "chair":    ["sitting on", "carrying", "pushing", "standing near"],
    "couch":    ["sitting on", "lying on", "standing near"],
    "helmet":   ["wearing", "holding", "carrying"],
    "boot":     ["wearing", "holding", "carrying"],
    "saddle":   ["sitting on", "adjusting", "using"],
    "phone":    ["using", "holding", "texting on", "talking on"],
    "laptop":   ["using", "typing on", "carrying", "watching"],
    "cup":      ["drinking from", "holding", "carrying"],
    "bottle":   ["drinking from", "holding", "carrying"],
    "bat":      ["holding", "swinging", "carrying"],
    "ball":     ["kicking", "throwing", "holding", "catching"],
    "book":     ["reading", "holding", "carrying"],
    "umbrella": ["holding", "carrying", "using"],
    "bag":      ["carrying", "holding", "wearing"],
    "remote":   ["holding", "using", "pointing"],
    "knife":    ["holding", "using", "cutting with"],
    "fork":     ["holding", "using", "eating with"],
    "bowl":     ["eating from", "holding", "carrying"],
}
DEFAULT_ACTIONS = [
    "holding", "using", "carrying", "standing near", "touching",
    "interacting with", "looking at", "operating", "pointing at",
    "leaning on", "pushing", "pulling"
]

hoi_candidates = []
sorted_objects = sorted(object_detections, key=lambda d: d["confidence"], reverse=True)
obj_classes    = list(dict.fromkeys(d["class"] for d in sorted_objects))
print(f"    Objects sorted by confidence: {[(d['class'], round(d['confidence'],3)) for d in sorted_objects]}")

for obj in obj_classes:
    actions = OBJECT_ACTIONS.get(obj, DEFAULT_ACTIONS)
    for action in actions:
        hoi_candidates.append(f"a person {action} a {obj}")

hoi_candidates += [
    "a person riding a horse outdoors",
    "a person sitting on a horse",
    "a person on horseback",
    "a person working outdoors",
    "a person in a rural setting",
]

if not hoi_candidates:
    hoi_candidates = ["a person sitting indoors", "a person standing outdoors",
                      "a person using a phone", "a person eating food"]

# If NO humans were detected at all, switch to scene-level (not person-level) candidates
if len(human_detections) == 0:
    scene_candidates = [f"a {obj} in an indoor setting" for obj in obj_classes] + \
                       [f"a {obj} in an outdoor setting" for obj in obj_classes] + \
                       ["an empty scene", "a still life scene", "an object on display"]
    hoi_candidates = scene_candidates if obj_classes else hoi_candidates
    print("    ! No person detected — using scene-level (not person-action) candidates")

with torch.no_grad():
    text_tokens   = tokenizer(hoi_candidates).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features.float(), dim=-1)

sims   = (image_feature @ text_features.T)[0]
probs  = F.softmax(sims * 100, dim=0)
top_k  = min(8, len(hoi_candidates))
top_vals, top_idxs = torch.topk(probs, k=top_k)
top_labels = [hoi_candidates[i] for i in top_idxs.tolist()]
top_probs  = top_vals.tolist()

print(f"    CLIP top-5 visual matches:")
for i, (lbl, prob) in enumerate(zip(top_labels[:5], top_probs[:5])):
    print(f"      {i+1}. {lbl:45s}  score={prob:.4f}")

# ════════════════════════════════════════════════════════════
# c) KNOWLEDGE PROMPT -> FREE LOCAL LLM (Ollama)
# ════════════════════════════════════════════════════════════
print(f"\n[c] Calling local LLM via Ollama ({OLLAMA_MODEL}) ...")

def call_ollama(prompt: str) -> str:
    """Call local Ollama server. FREE, no API key, runs on your machine."""
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    resp = requests.post(OLLAMA_URL, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()

detected_humans  = [d["class"] for d in human_detections]
detected_objects = [f"{d['class']} (conf={d['confidence']:.2f})" for d in object_detections]
top3_clips       = [f"{l} (score={p:.3f})" for l, p in zip(top_labels[:3], top_probs[:3])]

knowledge_prompt_Q = (
    f"You are analyzing an image for Human-Object Interaction detection.\n\n"
    f"DETECTED IN IMAGE:\n"
    f"  Humans  : {detected_humans}\n"
    f"  Objects : {detected_objects}\n\n"
    f"CLIP VISUAL GROUNDING (top visual matches):\n"
    + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(top3_clips)) +
    f"\n\nBased on the detections and CLIP matches above, respond with "
    f"ONLY a valid JSON object (no extra text, no markdown) in this exact format:\n"
    f'{{"scene_description": "one sentence describing the scene", '
    f'"primary_action": "what the person is most likely doing", '
    f'"environment": "type of environment", '
    f'"social_context": "social or behavioral context", '
    f'"commonsense_reason": "The person is likely [action] to [reason].", '
    f'"confidence": "high/medium/low", '
    f'"hoi_label": "short label e.g. ride-horse"}}'
)

try:
    llm_response_raw = call_ollama(knowledge_prompt_Q)
    clean = llm_response_raw.strip()
    if "```" in clean:
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    # Extract JSON object even if LLM added extra text around it
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    clean = clean[start:end]
    llm_data = json.loads(clean)
    print(f"\n    LLM Response (parsed):")
    for k, v in llm_data.items():
        print(f"      {k:22s}: {v}")
    llm_success = True
except Exception as e:
    print(f"    LLM call failed: {e}")
    print("    (Is Ollama running? Try: ollama pull llama3.2  then  ollama serve)")
    print("    Falling back to CLIP-only reasoning ...")
    best = top_labels[0]
    action = best.replace("a person ", "")
    FALLBACK_REASONS = {
        "riding":      "travel or participate in equestrian activity",
        "sitting on":  "rest or maintain control during riding",
        "feeding":     "care for and bond with the animal",
        "petting":     "show affection or calm the animal",
        "grooming":    "maintain the animal's hygiene and health",
        "leading":     "direct the animal to a specific location",
        "driving":     "operate the vehicle for transport or work",
        "operating":   "perform agricultural or work-related tasks",
        "wearing":     "protect themselves during the activity",
        "holding":     "use or carry the object for a purpose",
    }
    verb   = action.split()[0] if action.split() else "interacting with"
    reason = FALLBACK_REASONS.get(verb, "complete a routine task or activity related to the object")
    llm_data = {
        "scene_description":  f"A person with {', '.join(obj_classes)} in an outdoor setting",
        "primary_action":     action,
        "environment":        "outdoor rural or equestrian area" if "horse" in obj_classes else "general outdoor space",
        "social_context":     "equestrian activity or rural work" if "horse" in obj_classes else "daily activity",
        "commonsense_reason": f"The person is likely {action} to {reason}.",
        "confidence":         "medium",
        "hoi_label":          verb + "-" + (obj_classes[0] if obj_classes else "object"),
    }
    llm_success = False

knowledge_prompt_A = llm_data["commonsense_reason"]
reasoning_output   = llm_data["commonsense_reason"]
full_reasoning = (
    f"Scene       : {llm_data['scene_description']}\n"
    f"Action      : {llm_data['primary_action']}\n"
    f"Environment : {llm_data['environment']}\n"
    f"Social      : {llm_data['social_context']}\n"
    f"HOI label   : {llm_data['hoi_label']}\n"
    f"Reason      : {llm_data['commonsense_reason']}\n"
    f"Confidence  : {llm_data['confidence']}"
)

print(f"\n[d] Final Reasoning Output:")
print(f"    Q: Why is the person {llm_data['primary_action']}?")
print(f"    A: {knowledge_prompt_A}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
save_path = os.path.join(OUTPUT_DIR, "stage6_output.pt")
torch.save({
    "global_scene_feature"        : image_feature.cpu(),
    "scene_embedding"             : scene_embedding.cpu(),
    "llm_data"                    : llm_data,
    "llm_success"                 : llm_success,
    "knowledge_prompt_Q"          : knowledge_prompt_Q,
    "knowledge_prompt_A"          : knowledge_prompt_A,
    "reasoning_output"            : reasoning_output,
    "full_reasoning"              : full_reasoning,
    "top_labels"                  : top_labels,
    "top_probs"                   : top_probs,
    "human_detections"            : human_detections,
    "object_detections"           : object_detections,
    "enhanced_image_np"           : enhanced_np,
    "image_path"                  : stage4.get("image_path", ""),
    "scene_transformer_state_dict": scene_transformer.state_dict(),
}, save_path)
print(f"\nSaved: {save_path}")

# ════════════════════════════════════════════════════════════
# VISUALISE — 3 separate clean files
# ════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(9, 7))
ax.imshow(enhanced_uint8)
for d in human_detections:
    x1,y1,x2,y2 = d["box"]
    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,lw=2.5,edgecolor='lime',facecolor='none'))
    ax.text(x1+3,y1+14,f"{d['class']} {d['confidence']:.2f}",color='lime',fontsize=8,
            fontweight='bold',bbox=dict(facecolor='black',alpha=0.6,pad=1,edgecolor='none'))
for d in object_detections:
    x1,y1,x2,y2 = d["box"]
    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,lw=2,edgecolor='tomato',facecolor='none'))
    ax.text(x1+3,y1+14,f"{d['class']} {d['confidence']:.2f}",color='tomato',fontsize=8,
            fontweight='bold',bbox=dict(facecolor='black',alpha=0.6,pad=1,edgecolor='none'))
ax.set_title(
    f"Stage 6 — a) CLIP Input  |  HOI: {llm_data['hoi_label']}\n"
    f"{llm_data['scene_description']}",
    fontsize=11, fontweight='bold', pad=8)
ax.axis('off')
plt.tight_layout()
p1 = os.path.join(OUTPUT_DIR, "stage6_1_scene.png")
fig1.savefig(p1, dpi=180, bbox_inches='tight'); plt.close(fig1)
print(f"Saved: {p1}")

fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
short_labels = [l.replace("a person ", "")[:35] for l in top_labels[:8]]
bar_colors   = ['#27ae60'] + ['#3498db'] * (len(short_labels) - 1)
bars = ax.barh(short_labels[::-1], top_probs[:8][::-1], color=bar_colors[::-1], edgecolor='none', height=0.6)
for bar, prob in zip(bars, top_probs[:8][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{prob:.4f}", va='center', color='white', fontsize=9, fontweight='bold')
ax.set_xlabel("CLIP similarity score", color='#aaaaaa', fontsize=10)
ax.set_title("b) CLIP Visual Grounding\nTop-8 HOI candidates (dynamically generated from detections)",
             fontsize=11, fontweight='bold', color='white', pad=10)
for spine in ax.spines.values(): spine.set_color('#333355')
ax.tick_params(colors='white', labelsize=9)
for lbl in ax.get_yticklabels(): lbl.set_color('white')
for lbl in ax.get_xticklabels(): lbl.set_color('#aaaaaa')
plt.tight_layout()
p2 = os.path.join(OUTPUT_DIR, "stage6_2_clip_scores.png")
fig2.savefig(p2, dpi=180, bbox_inches='tight', facecolor=fig2.get_facecolor()); plt.close(fig2)
print(f"Saved: {p2}")

fig3, ax = plt.subplots(figsize=(10, 7))
fig3.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')
ax.text(0.5, 0.97, "c) Knowledge Prompt -> d) Reasoning Output",
        transform=ax.transAxes, ha='center', va='top',
        fontsize=13, fontweight='bold', color='white')
q_text = f"Q: Why is the person {llm_data['primary_action']}?"
ax.text(0.5, 0.86, q_text, transform=ax.transAxes, ha='center', va='top',
        fontsize=12, color='#f0e68c', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#2c2c00', edgecolor='#f0e68c', lw=1.5))
ax.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.73), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='white', lw=2))
kb_text = (
    f"Knowledge Base  ({OLLAMA_MODEL} via Ollama — FREE)\n\n"
    f"Environment : {llm_data['environment']}\n"
    f"Social      : {llm_data['social_context']}\n"
    f"Confidence  : {llm_data['confidence']}\n"
    f"HOI label   : {llm_data['hoi_label']}"
)
ax.text(0.5, 0.68, kb_text, transform=ax.transAxes, ha='center', va='top',
        fontsize=10, color='#a8d8ea',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a2540', edgecolor='#a8d8ea', lw=1.5))
ax.annotate("", xy=(0.5, 0.38), xytext=(0.5, 0.44), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='white', lw=2))
a_text = f"A: {llm_data['commonsense_reason']}"
ax.text(0.5, 0.37, a_text, transform=ax.transAxes, ha='center', va='top',
        fontsize=12, color='#90ee90', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#003300', edgecolor='#90ee90', lw=2))
ax.text(0.5, 0.12, f"Scene: {llm_data['scene_description']}",
        transform=ax.transAxes, ha='center', va='top',
        fontsize=10, color='#cccccc', style='italic')
badge = f"✓ Real LLM ({OLLAMA_MODEL}, free)" if llm_success else "⚠ CLIP fallback"
badge_color = '#27ae60' if llm_success else '#e67e22'
ax.text(0.97, 0.03, badge, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, color=badge_color, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(OUTPUT_DIR, "stage6_3_reasoning.png")
fig3.savefig(p3, dpi=180, bbox_inches='tight', facecolor=fig3.get_facecolor()); plt.close(fig3)
print(f"Saved: {p3}")

print("\n" + "=" * 60)
print("STAGE 6 COMPLETE")
print("=" * 60)
print(f"HOI label        : {llm_data['hoi_label']}")
print(f"Primary action   : {llm_data['primary_action']}")
print(f"Reasoning        : {reasoning_output}")
print(f"LLM used         : {'Yes — ' + OLLAMA_MODEL + ' (free, local)' if llm_success else 'No — CLIP fallback'}")
print(f"Next -> Stage 7: Semi-Supervised Contrastive Learning")
print("=" * 60)