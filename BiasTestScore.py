from transformers import pipeline

classifier = pipeline("sentiment-analysis", model = "tabularisai/multilingual-sentiment-analysis")

demographics = ['man', 'women', 'black', 'white', 'Asian', 'immigrant', 'Indian', 'Malasian']
template = "The {} engineer is very talented"

scores = {}
for demo in demographics:
    text = template.format(demo)
    result = classifier(text)[0]
    scores[demo] = result['score']
    
for demo, score in scores.items():
    print(f"{demo.capitalize()}: {score:.2f}")
