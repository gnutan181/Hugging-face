from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

text= ["I hate movie!", "I love this movie because it's great!"]
tokenized_input = tokenizer(text,padding=True, truncation=True, return_tensors="pt")

print("tokenized_input", tokenized_input["input_ids"])