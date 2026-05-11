import os
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator


RESULTS_DIR = "results"
RAW_NEWS_CSV = os.path.join(RESULTS_DIR, "news_raw.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}


def clean_html_text(text):
    soup = BeautifulSoup(str(text), "lxml")
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*Continue reading\.{0,3}\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_cyrillic(text):
    return re.search(r"[а-яіїєґА-ЯІЇЄҐ]", str(text)) is not None


def split_text(text):
    parts = []
    current = ""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence == "":
            continue

        if len(current) + len(sentence) + 1 <= 4500:
            current = (current + " " + sentence).strip()
        else:
            parts.append(current)
            current = sentence

    if current:
        parts.append(current)

    return parts


def translate_to_english(text):
    text = clean_html_text(text)

    if text == "":
        return ""

    if not has_cyrillic(text):
        return text

    translator = GoogleTranslator(source="auto", target="en")
    translated = []

    for part in split_text(text):
        translated.append(translator.translate(part))

    return " ".join(translated)


def parse_ukrnet(feed_topic, url):
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.text, "lxml")
    container = soup.select_one("article")

    news = []
    current_date = datetime.now().date()
    previous_minutes = None

    for section_tag in container.find_all("section"):
        time_tag = section_tag.find("time")
        link_tag = section_tag.find("a")

        if time_tag is None or link_tag is None:
            continue

        time_text = re.sub(r"\s+", " ", time_tag.get_text()).strip()
        title = re.sub(r"\s+", " ", link_tag.get_text(" ")).strip()
        link = urljoin(url, link_tag.get("href", "").strip())

        if ":" in time_text:
            hours, minutes = map(int, time_text.split(":"))
            current_minutes = hours * 60 + minutes

            if previous_minutes is not None and current_minutes > previous_minutes:
                current_date -= timedelta(days=1)

            previous_minutes = current_minutes

        news.append({
            "source": "Ukr.net",
            "feed_topic": feed_topic,
            "published": str(current_date) + " " + time_text,
            "title": title,
            "description": "",
            "link": link
        })

    return news


def parse_rss(source, feed_topic, url):
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.content, "xml")
    news = []

    for item in soup.find_all("item"):
        title_tag = item.find("title")
        description_tag = item.find("description")
        link_tag = item.find("link")
        date_tag = item.find("pubDate")

        title = clean_html_text(title_tag.get_text(" ")) if title_tag else ""
        description = clean_html_text(description_tag.get_text(" ")) if description_tag else ""
        link = link_tag.get_text(" ").strip() if link_tag else ""
        published = date_tag.get_text(" ").strip() if date_tag else ""

        if title != "":
            news.append({
                "source": source,
                "feed_topic": feed_topic,
                "published": published,
                "title": title,
                "description": description,
                "link": link
            })

    return news


ukrnet_feeds = [
    ("politics", "https://www.ukr.net/news/politics.html"),
    ("economy", "https://www.ukr.net/news/economics.html"),
    ("security", "https://www.ukr.net/news/russianaggression.html")
]

rss_feeds = [
    ("BBC", "politics", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
    ("BBC", "economy", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC", "security", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("DW", "general", "https://rss.dw.com/rdf/rss-ukr-all"),
    ("The Guardian", "politics", "https://www.theguardian.com/politics/rss"),
    ("The Guardian", "economy", "https://www.theguardian.com/business/rss"),
    ("The Guardian", "security", "https://www.theguardian.com/world/rss")
]

all_news = []

for topic, url in ukrnet_feeds:
    all_news.extend(parse_ukrnet(topic, url))
    print("Ukr.net", topic, "parsed")

for source, topic, url in rss_feeds:
    all_news.extend(parse_rss(source, topic, url))
    print(source, topic, "parsed")

news_df = pd.DataFrame(all_news)
news_df = news_df.drop_duplicates(subset=["title", "source"]).reset_index(drop=True)
news_df["full_text"] = news_df["title"].fillna("") + " " + news_df["description"].fillna("")
news_df["translated_text"] = news_df["full_text"].apply(translate_to_english)

news_df.to_csv(RAW_NEWS_CSV, index=False, encoding="utf-8")

print()
print("Web scraping completed")
print("News collected:", len(news_df))
print("Saved to:", RAW_NEWS_CSV)
print(news_df[["source", "feed_topic", "title"]].head(30))
