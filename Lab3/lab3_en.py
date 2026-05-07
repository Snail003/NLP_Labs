import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


FILE_NAME = "Mcdonald.csv"


def clear_review(text):
    text = str(text).lower()
    text = text.replace("�", " ")
    text = re.sub(r"\bwon['’ʼ]t\b", "will not", text)
    text = re.sub(r"\bcan['’ʼ]t\b", "can not", text)
    text = re.sub(r"n['’ʼ]t\b", " not", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


reviews = pd.read_csv(FILE_NAME)

reviews = reviews[["review", "sentiment"]].copy()

reviews["sentiment"] = reviews["sentiment"].str.lower().str.strip()
reviews = reviews[reviews["sentiment"].isin(["negative", "neutral", "positive"])].copy()

reviews["review_clean"] = reviews["review"].apply(clear_review)
reviews = reviews[reviews["review_clean"].str.len() > 3].copy()

train, test = train_test_split(
    reviews,
    test_size=0.25,
    random_state=42,
    stratify=reviews["sentiment"]
)

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    min_df=2
)

x_train = vectorizer.fit_transform(train["review_clean"])
x_test = vectorizer.transform(test["review_clean"])

y_train = train["sentiment"]
y_test = test["sentiment"]

model = LinearSVC(
    class_weight="balanced",
    max_iter=5000
)

model.fit(x_train, y_train)


def predict_review(text):
    text_clean = clear_review(text)
    text_vector = vectorizer.transform([text_clean])
    return model.predict(text_vector)[0]


predicted = model.predict(x_test)

print("Кількість відгуків:", len(reviews))

print("\nКласи у навчальній вибірці:")
print(train["sentiment"].value_counts())

print("\nКласи у тестовій вибірці:")
print(test["sentiment"].value_counts())

print("\nМатриця помилок:")
print(confusion_matrix(y_test, predicted, labels=["negative", "neutral", "positive"]))

print("\nЗвіт класифікації:")
print(classification_report(y_test, predicted, digits=3, zero_division=0))

result = test[["review", "sentiment"]].copy()
result["predicted_sentiment"] = predicted
result.to_csv("mcdonald_result.csv", index=False, encoding="utf-8")

examples = [
    "The food was fresh and the workers were very friendly.",
    "The service was slow and my order was completely wrong.",
    "It was a normal visit, nothing special happened.",
    "Normal food and prices.",
    "The restaurant was dirty and I will not come back.",
    "The food was okay, but nothing special.",
    "Average experience overall.",
    "I won't recommend this place."
]

print("\nВласні приклади:")

for item in examples:
    answer = predict_review(item)
    print(answer, "-", item)
