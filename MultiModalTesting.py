from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# STEP 1: Load a Pretrained CLIP Model and Processor
# -----------------------------
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# -----------------------------
# STEP 2: Prepare an Image and Candidate Text Prompts
# -----------------------------
image_path = "../image/images.jpeg" 
image = Image.open(image_path).convert("RGB")

candidate_texts = [
    "a puppy dogs",
    "a cats",
    "a bird",
    "a person",
    "a robot dog"
    "a bowl of fruit",
]

# -----------------------------
# STEP 3: Tokenize and Encode Both Image and Text
# -----------------------------
inputs = processor(
    text=candidate_texts,
    images=image,
    return_tensors="pt",
    padding=True
)

# -----------------------------
# STEP 4: Forward Pass Through CLIP
# -----------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

# -----------------------------
# STEP 5: Inspect the Results
# -----------------------------
for text, prob in zip(candidate_texts, probs[0]):
    print(f"'{text}' --> {prob.item():.4f}")

# Identify the top label
max_idx = probs[0].argmax().item()
predicted_label = candidate_texts[max_idx]
print(f"\nPredicted label: '{predicted_label}' with confidence {probs[0, max_idx].item():.4f}")