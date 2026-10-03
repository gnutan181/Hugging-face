from transformers import pipeline


generator = pipeline("text-generation", model="gpt2")

prompt = "This is the story of how AI models started taking over the world by gaining access"


low_temperature_output = generator(prompt, truncation=True, num_return_sequences=1, temperature=0.2)
high_temperature_output = generator(prompt, truncation=True, num_return_sequences=1, temperature=1.0)
print("Low Temperature Output:",low_temperature_output)
# print(low_temperature_output[0]['generated_text'])
# print("---")
# print(high_temperature_output[0]['generated_text'])