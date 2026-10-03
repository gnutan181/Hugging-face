from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "The future of AI is",
    "In the world of science and advanced technology",
    "Once humanity reaches Mars, we will",
]

for prompt in prompts:
  for temp in [0.1, 0.5, 0.8, 1.0, 1.2]:
    output = generator(prompt, truncation=True, temperature=temp, max_length=150)
    print(f"Prompt: {prompt}, Temperature: {temp}")
    print(output[0]['generated_text'])
    print("\n")