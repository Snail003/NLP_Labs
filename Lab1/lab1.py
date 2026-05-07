import re
from datetime import datetime, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


URL = "https://www.ukr.net/news/main.html"
TXT_FILE = "output.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

MONTHS = {
    1: "січня",
    2: "лютого",
    3: "березня",
    4: "квітня",
    5: "травня",
    6: "червня",
    7: "липня",
    8: "серпня",
    9: "вересня",
    10: "жовтня",
    11: "листопада",
    12: "грудня",
}


response = requests.get(URL, headers=HEADERS)
response.raise_for_status()

html = response.text

soup = BeautifulSoup(html, "lxml")

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
    href = link_tag.get("href", "").strip()

    link = urljoin(URL, href)

    hours, minutes = map(int, time_text.split(":"))
    current_minutes = hours * 60 + minutes

    if previous_minutes is not None and current_minutes > previous_minutes:
        current_date -= timedelta(days=1)

    date_text = f"{current_date.day} {MONTHS[current_date.month]}"

    item = {
        "date": date_text,
        "time": time_text,
        "title": title,
        "link": link,
    }

    news.append(item)

    previous_minutes = current_minutes


with open(TXT_FILE, "w", encoding="utf-8") as file:
    for item in news:
        file.write(f"Дата: {item['date']}\n")
        file.write(f"Час: {item['time']}\n")
        file.write(f"Заголовок: {item['title']}\n")
        file.write(f"Посилання: {item['link']}\n")
        file.write("-" * 80 + "\n")


print("Знайдено новин:", len(news))