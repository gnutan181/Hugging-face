from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch 


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sentence = "I hate this movie!"
input = tokenizer(sentence, return_tensors="pt")

logits = model(**input).logits
print("loggits",logits)
predicted_class_id = logits.argmax().item()

print("prdicted",predicted_class_id)  # Output: 1 (positive sentiment)

print("POSITIVE" if predicted_class_id == 1 else "NEGATIVE")  # Output: POSITIVE