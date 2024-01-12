import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import models

# Load the dataset
file_path = 'HIV_dataset.json'
with open(file_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

questions = list(dataset.keys())
responses = list(dataset.values())

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(questions)
model = models.select_ml_algorithm()

model.fit(X_vectorized, responses)

y_pred = model.predict(X_vectorized)

accuracy = accuracy_score(responses, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
precision = precision_score(responses, y_pred, average='weighted',zero_division=1)
recall = recall_score(responses, y_pred, average='weighted',zero_division=1)
f1 = f1_score(responses, y_pred, average='weighted')

print("Other Evaluation Metrics:")
print(f"Precision: {precision*100:.4f}")
print(f"Recall: {recall*100:.4f}")
print(f"F1-Score: {f1*100:.4f}")

