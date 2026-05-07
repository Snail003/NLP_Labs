import re
from collections import Counter
from pathlib import Path

import spacy


INPUT_FILE = Path("ukrainka-lesia-lisova-pisnia.txt")
LANGUAGE = "uk"
TOP_KEYWORDS = 10

RESULTS_DIR = Path("results")

FILTERED_FILE = RESULTS_DIR / "01_filtered_text.txt"
NORMALIZED_FILE = RESULTS_DIR / "02_normalized_text.txt"
TOKENS_FILE = RESULTS_DIR / "03_tokens.txt"
WITHOUT_STOPWORDS_FILE = RESULTS_DIR / "04_without_stopwords.txt"
LEMMAS_FILE = RESULTS_DIR / "05_lemmas.txt"
KEYWORDS_FILE = RESULTS_DIR / "06_keywords.txt"


RESULTS_DIR.mkdir(exist_ok=True)

if LANGUAGE == "uk":
    nlp = spacy.load("uk_core_news_sm")
else:
    nlp = spacy.load("en_core_web_sm")


text = INPUT_FILE.read_text(encoding="utf-8")


filtered_text = re.sub(r"http\S+|www\.\S+", " ", text)
filtered_text = re.sub(r"\d+", " ", filtered_text)
filtered_text = re.sub(r"[^A-Za-zА-Яа-яІіЇїЄєҐґ'ʼ’\s]", " ", filtered_text)
filtered_text = re.sub(r"\s+", " ", filtered_text)
filtered_text = filtered_text.strip()

FILTERED_FILE.write_text(filtered_text, encoding="utf-8")


normalized_text = filtered_text.lower()
normalized_text = normalized_text.replace("’", "'")
normalized_text = normalized_text.replace("ʼ", "'")

NORMALIZED_FILE.write_text(normalized_text, encoding="utf-8")


doc = nlp(normalized_text)

tokens = []

for token in doc:
    if token.is_alpha:
        tokens.append(token)

token_words = []

for token in tokens:
    token_words.append(token.text)

TOKENS_FILE.write_text("\n".join(token_words), encoding="utf-8")


tokens_without_stopwords = []

for token in tokens:
    if token.is_stop:
        continue

    if len(token.text) < 3:
        continue

    tokens_without_stopwords.append(token)

words_without_stopwords = []

for token in tokens_without_stopwords:
    words_without_stopwords.append(token.text)

WITHOUT_STOPWORDS_FILE.write_text("\n".join(words_without_stopwords), encoding="utf-8")


lemmas = []

for token in tokens_without_stopwords:
    if token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "ADJ":
        lemma = token.lemma_.lower()
        lemmas.append(lemma)

LEMMAS_FILE.write_text("\n".join(lemmas), encoding="utf-8")


keywords = Counter(lemmas).most_common(TOP_KEYWORDS)

keyword_lines = []

for word, count in keywords:
    keyword_lines.append(f"{word}: {count}")

KEYWORDS_FILE.write_text("\n".join(keyword_lines), encoding="utf-8")


print("Обробку завершено.")