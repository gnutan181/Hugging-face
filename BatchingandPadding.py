from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text= ["I hate this movie!", "I love this movie!"]
tokenized_input = tokenizer(text,padding=True, truncation=True, return_tensors="pt")

print("tokenized_input", tokenized_input["input_ids"])