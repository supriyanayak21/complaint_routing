import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("dataset.csv")
X = df["text"]
y = df["priority"]
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors,
    y,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nMODEL EVALUATION:\n")
print(classification_report(y_test, predictions))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/priority_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")


print("\nModel trained and saved successfully!")