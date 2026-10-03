from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentences = ["I am learning Machine learning",
             "I love ML and HuggingFace",
             "I am not great at learning human science",
             "Transformers are great at understanding natural language"]

input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
logits = model(**input).logits

predicted_class_ids = torch.argmax(logits,dim=-1).tolist()
print("logits", torch.argmax(logits))  # Output: tensor([1, 1, 0, 1]) (predicted class IDs for each sentence)
print("predicted_class_ids", predicted_class_ids)
labels = ["NEGATIVE", "POSITIVE"]
for sentences, predicted_class_id in zip(sentences,predicted_class_ids):
    print(f"Sentence: {sentences} | Sentiment: {labels[predicted_class_id]}")
