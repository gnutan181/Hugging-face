from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input text for explainability testing
input_text = "Playwright is a UI Testing tool"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

outputs = model(**inputs, output_attentions=True)

# Extract attention weights
attentions = outputs.attentions

# Select attention from the last layer for visualization
last_layer_attention = attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)

# Aggregate attention across heads (mean)
avg_attention = last_layer_attention.mean(dim=1).squeeze(0).detach().numpy()

# Decode tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

# Plot attention heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
plt.title("Attention Heatmap (Last Layer)")
plt.xlabel("Input Tokens")
plt.ylabel("Input Tokens")
plt.xticks(rotation=90)
plt.show()

# Explainability using attention
print("Tokens:", tokens)
print("Aggregated attention scores:", avg_attention)