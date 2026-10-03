from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "I am learning Machine Learning"

encoded_input  = tokenizer(sentence)
print("Encoded Input:", encoded_input)
print("Tokens:" , tokenizer.convert_ids_to_tokens(encoded_input["input_ids"]))
