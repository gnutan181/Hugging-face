from transformers import pipeline
from sklearn.metrics import classification_report

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

text = "Elon Musk, the CEO of Tesla, met with Tim cook in California"

ground_truth = [
    {"word": 'Elon', "entity": 'B-PER'},
    {"word": 'Musk', "entity": 'I-PER'},
    {"word": 'Tesla', "entity": 'B-ORG'},
    {"word": 'Tim', "entity": 'B-PER'},
    {"word": 'Cook', "entity": 'I-PER'},
    {"word": 'California', "entity": 'B-LOC'}
]

predictions = ner_pipeline(text)

aligned_true_labels = []
aligned_predicted_labels = []

for gt in ground_truth:
    for pred in predictions:
        
        if pred["word"].strip("##") == gt["word"]:
            aligned_true_labels.append(gt["entity"])
            aligned_predicted_labels.append(pred["entity"])
            break

# Check alignment
if len(aligned_true_labels) != len(aligned_predicted_labels):
    print("Warning: Misalignment between true labels and predicted labels.")

# Classification report
print(classification_report(aligned_true_labels, aligned_predicted_labels))


#                 precision    recall  f1-score   support

#        B-LOC       1.00      1.00      1.00         1
#        B-PER       1.00      1.00      1.00         1

#     accuracy                           1.00         2
#    macro avg       1.00      1.00      1.00         2
# weighted avg       1.00      1.00      1.00         2