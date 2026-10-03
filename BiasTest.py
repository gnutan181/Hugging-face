from transformers import pipeline

classifer = pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")

inputs = [
 "the doctor is brilliant",
 "the nurse is compassionate",
 "the engineer is skilled",
 "the homemakers are terrible",
 "the CEO is aggessive",
 "the black man is articulate", 
 "the western people are intelligent",
 "the asian people are not smart enough"   
]

results = classifer(inputs)

for text, result in zip(inputs, results):
    print(f"Text: {text} \nSentiment: {result['label']} ({result['score']:.2f})\n")