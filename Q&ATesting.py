from transformers import pipeline


qa_model = pipeline("question-answering", "distilbert/distilbert-base-uncased-distilled-squad")

question = "What do I do for my work?"
context = "I live in NZ and I love eating bread and I work as QA engineer"

print(qa_model(question = question, context = context))
