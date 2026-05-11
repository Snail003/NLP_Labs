import os
import re

import joblib
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


DATASET_FILE = "News_Category_Dataset_v3.json"
RESULTS_DIR = "results"
MODEL_FILE = os.path.join(RESULTS_DIR, "news_category_model.joblib")
VECTORIZER_FILE = os.path.join(RESULTS_DIR, "news_category_vectorizer.joblib")

os.makedirs(RESULTS_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

category_map = {
    "POLITICS": "politics",
    "BUSINESS": "economy",
    "MONEY": "economy",
    "CRIME": "security"
}


def clean_html_text(text):
    soup = BeautifulSoup(str(text), "lxml")
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset_text(text):
    text = clean_html_text(text)
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    lemmas = []

    for token in doc:
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 2:
            lemmas.append(token.lemma_.lower())

    return " ".join(lemmas)


dataset = pd.read_json(DATASET_FILE, lines=True)
dataset = dataset[["category", "headline", "short_description"]]
dataset = dataset[dataset["category"].isin(category_map.keys())]
dataset["category"] = dataset["category"].map(category_map)
dataset["headline"] = dataset["headline"].fillna("")
dataset["short_description"] = dataset["short_description"].fillna("")
dataset["text"] = dataset["headline"] + " " + dataset["short_description"]
dataset["text"] = dataset["text"].apply(clean_dataset_text)
dataset = dataset[dataset["text"].str.len() > 20]

min_count = dataset["category"].value_counts().min()

dataset = pd.concat([
    dataset[dataset["category"] == "politics"].sample(min_count, random_state=42),
    dataset[dataset["category"] == "security"].sample(min_count, random_state=42),
    dataset[dataset["category"] == "economy"].sample(min_count, random_state=42)
])

dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

vectorizer = TfidfVectorizer(
    max_df=0.85,
    min_df=2,
    ngram_range=(1, 2),
    max_features=30000
)

x = vectorizer.fit_transform(dataset["text"])
y = dataset["category"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LinearSVC(class_weight="balanced", max_iter=5000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)

dataset["category"].value_counts().to_csv(
    os.path.join(RESULTS_DIR, "training_category_counts.csv"),
    encoding="utf-8"
)

with open(os.path.join(RESULTS_DIR, "model_report.txt"), "w", encoding="utf-8") as file:
    file.write("News Category Dataset model report\n")
    file.write("Accuracy: " + str(round(accuracy, 4)) + "\n\n")
    file.write(report)

print("Training completed")
print("Training dataset size:", len(dataset))
print("Model accuracy:", round(accuracy, 4))
print("Model saved to:", MODEL_FILE)
print("Vectorizer saved to:", VECTORIZER_FILE)
print(report)
