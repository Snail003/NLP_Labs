import re
import requests
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


DATASET_FILE = "News_Category_Dataset_v3.json"

nlp = spacy.load("en_core_web_sm")

headers = {
    "User-Agent": "Mozilla/5.0"
}

category_map = {
    "BUSINESS": "economic",
    "POLITICS": "political",
    "WORLD NEWS": "political",
    "WELLNESS": "social",
    "HEALTHY LIVING": "social",
    "EDUCATION": "social",
    "SPORTS": "sport"
}

feeds = [
    ["economic", "https://feeds.bbci.co.uk/news/business/economy/rss.xml"],

    ["political", "https://feeds.bbci.co.uk/news/politics/rss.xml"],

    ["social", "https://feeds.bbci.co.uk/news/health/rss.xml"],
    ["social", "https://feeds.bbci.co.uk/news/education/rss.xml"],

    ["sport", "https://feeds.bbci.co.uk/sport/rss.xml"],
    ["sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"],
    ["sport", "https://feeds.bbci.co.uk/sport/tennis/rss.xml"],
    ["sport", "https://feeds.bbci.co.uk/sport/cricket/rss.xml"],
    ["sport", "https://feeds.bbci.co.uk/sport/rugby-union/rss.xml"],
    ["sport", "https://feeds.bbci.co.uk/sport/formula1/rss.xml"]
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text):
    doc = nlp(text)
    lemmas = []

    for token in doc:
        if not token.is_stop and token.is_alpha and len(token.lemma_) > 2:
            lemmas.append(token.lemma_)

    return " ".join(lemmas)


dataset = pd.read_json(DATASET_FILE, lines=True)

dataset = dataset[["category", "headline", "short_description"]].copy()
dataset = dataset[dataset["category"].isin(category_map.keys())].copy()

dataset["category"] = dataset["category"].map(category_map)
dataset["headline"] = dataset["headline"].fillna("")
dataset["short_description"] = dataset["short_description"].fillna("")
dataset["text"] = dataset["headline"] + " " + dataset["short_description"]
dataset["text"] = dataset["text"].apply(clean_text)
dataset = dataset[dataset["text"].str.len() > 20].copy()

min_count = dataset["category"].value_counts().min()

dataset = pd.concat([
    dataset[dataset["category"] == "economic"].sample(min_count, random_state=42),
    dataset[dataset["category"] == "political"].sample(min_count, random_state=42),
    dataset[dataset["category"] == "social"].sample(min_count, random_state=42),
    dataset[dataset["category"] == "sport"].sample(min_count, random_state=42)
])

dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print("Навчальний датасет News Category Dataset")
print(dataset[["headline", "category"]])
print()

print("Кількість новин у навчальному наборі")
print(len(dataset))
print()

print("Розподіл категорій у навчальному наборі")
print(dataset["category"].value_counts())
print()

dataset["text"] = dataset["text"].apply(lemmatize_text)

vectorizer = TfidfVectorizer(
    max_df=0.8,
    min_df=2,
    ngram_range=(1, 2),
    max_features=30000
)

X = vectorizer.fit_transform(dataset["text"])
y = dataset["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LinearSVC(
    class_weight="balanced",
    max_iter=5000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Ефективність моделі на News Category Dataset")
print("Accuracy:", round(accuracy, 4))
print()

print(classification_report(y_test, y_pred, zero_division=0))

bbc_titles = []
bbc_descriptions = []
bbc_texts = []
bbc_categories = []
bbc_links = []

for category, url in feeds:
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "xml")
    items = soup.find_all("item")

    for item in items:
        title_tag = item.find("title")
        description_tag = item.find("description")
        link_tag = item.find("link")

        title = title_tag.text if title_tag else ""
        description = description_tag.text if description_tag else ""
        link = link_tag.text if link_tag else ""

        text = title + " " + description
        text = clean_text(text)
        text = lemmatize_text(text)

        bbc_titles.append(title)
        bbc_descriptions.append(description)
        bbc_texts.append(text)
        bbc_categories.append(category)
        bbc_links.append(link)

df_bbc = pd.DataFrame({
    "title": bbc_titles,
    "description": bbc_descriptions,
    "link": bbc_links,
    "text": bbc_texts,
    "category": bbc_categories
})

df_bbc = df_bbc.drop_duplicates(subset=["title"])

print()
print("Зібрані новини BBC")
print(df_bbc[["title", "category"]])
print()

print("Розподіл новин BBC за початковими сферами")
print(df_bbc["category"].value_counts())

X_bbc = vectorizer.transform(df_bbc["text"])
df_bbc["predicted_category"] = model.predict(X_bbc)

bbc_accuracy = accuracy_score(df_bbc["category"], df_bbc["predicted_category"])

print()
print("Ефективність моделі на BBC RSS")
print("Accuracy:", round(bbc_accuracy, 4))
print()

print(classification_report(
    df_bbc["category"],
    df_bbc["predicted_category"],
    zero_division=0
))

print()
print("Результат класифікації BBC новин")
print(df_bbc[["title", "category", "predicted_category"]])
print()

print("Розподіл визначених сфер BBC")
print(df_bbc["predicted_category"].value_counts())

df_bbc.to_csv("bbc_news_result.csv", index=False, encoding="utf-8")


def predict_category(text):
    text = clean_text(text)
    text = lemmatize_text(text)

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    return prediction[0]


examples = [
    "The government discussed new laws after the election.",
    "Doctors warn about a new disease spreading among children.",
    "The football team won the final match after penalties.",
    "The company reported higher profits after strong sales."
]

print()
print("Власні приклади:")

for sample_text in examples:
    predicted_category = predict_category(sample_text)
    print(sample_text, "=>", predicted_category)