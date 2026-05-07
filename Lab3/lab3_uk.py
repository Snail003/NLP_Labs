import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC


FILE_NAME = "ua_reviews.csv"


def clear_review(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zа-яіїєґ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def category_to_sentiment(category):
    category = str(category).strip()

    if category == "Gratitude / Positive Feedback":
        return "positive"

    if category == "Complaint / Dissatisfaction":
        return "negative"

    return "neutral"


reviews = pd.read_csv(FILE_NAME)

reviews = reviews[["content", "final_category", "split"]].copy()

reviews["target"] = reviews["final_category"].apply(category_to_sentiment)
reviews["text_clean"] = reviews["content"].apply(clear_review)

reviews = reviews[reviews["text_clean"].str.len() > 3].copy()

train = reviews[reviews["split"] == "train"].copy()
test = reviews[reviews["split"] == "test"].copy()

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2
)

x_train = vectorizer.fit_transform(train["text_clean"])
x_test = vectorizer.transform(test["text_clean"])

y_train = train["target"]
y_test = test["target"]

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
print(train["target"].value_counts())

print("\nКласи у тестовій вибірці:")
print(test["target"].value_counts())

print("\nМатриця помилок:")
print(confusion_matrix(y_test, predicted, labels=["negative", "neutral", "positive"]))

print("\nЗвіт класифікації:")
print(classification_report(y_test, predicted, digits=3, zero_division=0))

result = test[["content", "final_category", "target"]].copy()
result["predicted_target"] = predicted
result.to_csv("ua_reviews_result.csv", index=False, encoding="utf-8")

examples = [
    "Працівники швидко допомогли і ввічливо пояснили всі деталі.",
    "Не рекомендую цей сервіс, черга була величезна і ніхто не допоміг.",
    "Послугу отримав у зазначений день, усе було стандартно.",
    "Сервіс нормальний, але нічого особливого не було.",
    "Дуже задоволений обслуговуванням, усе сподобалося.",
    "Просто звичайний сервіс.",
    "Нормальна послуга, без особливих переваг або недоліків.",
    "Усе було стандартно, нічого поганого не сталося."
]

print("\nВласні приклади:")

for item in examples:
    answer = predict_review(item)
    print(answer, "-", item)
