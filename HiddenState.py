

from transformers import AutoModel, AutoTokenizer

checkpoint  = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

Sentence = "I am learning Machine Learning"
input = tokenizer(Sentence, return_tensors="pt")
print("Encoded Input:", input)

output = model(**input)
print("Output:", output.last_hidden_state.shape)