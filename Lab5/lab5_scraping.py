import re
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

WORKUA_URLS = [
    "https://www.work.ua/jobs-remote-%D0%B4%D0%B0%D1%82%D0%B0-%D0%B0%D0%BD%D0%B0%D0%BB%D1%96%D1%82%D0%B8%D0%BA/",
    "https://www.work.ua/jobs-data+analyst/",
    "https://www.work.ua/jobs-%D0%B0%D0%BD%D0%B0%D0%BB%D1%96%D1%82%D0%B8%D0%BA+%D0%B4%D0%B0%D0%BD%D0%B8%D1%85/"
]

DJINNI_URL = "https://djinni.co/jobs/?primary_keyword=Data+Analyst"
LINKEDIN_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

WORKUA_PAGES = 3
LINKEDIN_PAGES = 3
DJINNI_PAGES = 3
DOU_REVIEWS_LIMIT_PER_COMPANY = 5

JOB_PATTERN = re.compile(
    r"data\s*analyst|дата\s*-?\s*аналітик|аналітик\s+даних|bi\s*analyst|business\s*data\s*analyst",
    re.IGNORECASE
)


def clean_text(text):
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_page(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=20)
    if response.status_code == 200:
        return response.text
    return ""

def get_text(tag):
    if tag is None:
        return ""
    return clean_text(tag.get_text(" ", strip=True))

def add_page(url, page):
    if page == 1:
        return url
    if "?" in url:
        return url + f"&page={page}"
    return url + f"?page={page}"

def company_key(company):
    company = clean_text(company).lower()
    company = company.replace("&", " and ")
    company = re.sub(r"\b(llc|ltd|inc|corp|company|group|тов|фоп|компанія)\b", " ", company)
    company = re.sub(r"[^a-zа-яіїєґ0-9]+", "-", company)
    company = re.sub(r"-+", "-", company)
    return company.strip("-")

def split_text(text, max_len=900):
    text = clean_text(text)
    parts = []

    while len(text) > max_len:
        cut = max(text.rfind(". ", 0, max_len), text.rfind(", ", 0, max_len), text.rfind(" ", 0, max_len))

        if cut < 300:
            cut = max_len

        parts.append(clean_text(text[:cut]))
        text = clean_text(text[cut:])

    if text:
        parts.append(text)

    return parts

def translate_text(text):
    text = clean_text(text)
    translated_parts = []

    for part in split_text(text, 900):
        translated = GoogleTranslator(source="auto", target="en").translate(part)
        translated_parts.append(clean_text(translated))

    return clean_text(" ".join(translated_parts))


def normalize_company(company):
    company = str(company)
    company = company.replace("Більше про компанію", "")
    company = company.replace("Про компанію", "")
    company = re.sub(r"^Робота в\s+", "", company, flags=re.IGNORECASE)
    company = re.sub(r"\s+на\s+Work\.ua.*$", "", company, flags=re.IGNORECASE)
    company = re.sub(r"\s+вакансії.*$", "", company, flags=re.IGNORECASE)
    return clean_text(company.strip(" «»\"'.,;:-–—"))


def normalize_location(location):
    location = str(location)
    location = location.replace("Місце роботи", "")
    location = location.replace("Адреса роботи", "")
    location = location.replace("Адреса", "")
    location = location.replace("Місце", "")
    location = re.sub(r"Показати на карті.*$", "", location)
    location = re.sub(r"На мапі.*$", "", location)
    location = re.sub(r"\d+[,.]?\d*\s*км від центру.*$", "", location)
    location = clean_text(location)

    if "Дистанційно" in location:
        return "Дистанційно"

    if "Full Remote" in location:
        return "Full Remote"

    if "Remote" in location:
        return "Remote"

    first_part = location.split(",")[0].strip()

    cities = [
        "Київ", "Львів", "Дніпро", "Харків", "Одеса", "Вінниця", "Запоріжжя",
        "Kyiv", "Kiev", "Lviv", "Dnipro", "Kharkiv", "Odesa", "Odessa",
        "Warsaw", "Krakow", "Berlin"
    ]

    for city in cities:
        if first_part.lower() == city.lower():
            return city

    for city in cities:
        if re.search(rf"\b{re.escape(city)}\b", location, re.IGNORECASE):
            return city

    return first_part if first_part else "Не вказано"


def get_linkedin_job_id(link):
    numbers = re.findall(r"\d{6,}", link)
    return numbers[-1] if numbers else ""


jobs = []
used_links = set()

for search_url in WORKUA_URLS:
    for page in range(1, WORKUA_PAGES + 1):
        html = get_page(add_page(search_url, page))
        soup = BeautifulSoup(html, "html.parser")
        links = soup.find_all("a", href=re.compile(r"^/jobs/\d+/"))

        for link_tag in links:
            link = urljoin("https://www.work.ua", link_tag.get("href", ""))

            if link in used_links:
                continue

            used_links.add(link)
            detail_soup = BeautifulSoup(get_page(link), "html.parser")

            title = get_text(detail_soup.find("h1"))
            description = get_text(
                detail_soup.find(id="job-description")
                or detail_soup.select_one(".job-description")
                or detail_soup.find("main")
            )
            company_tag = detail_soup.select_one('a[href^="/jobs/by-company/"] span.strong-500')
            company = normalize_company(get_text(company_tag))

            if not company:
                company = "Unknown"

            location_tag = detail_soup.find("span", attrs={"title": "Місце роботи"}) or detail_soup.find("span", attrs={"title": "Адреса роботи"})

            if location_tag:
                location = normalize_location(get_text(location_tag) or get_text(location_tag.parent))
            elif "Дистанційно" in get_text(detail_soup):
                location = "Дистанційно"
            else:
                location = "Не вказано"

            if JOB_PATTERN.search(title + " " + description):
                jobs.append({
                    "platform": "Work.ua",
                    "title": title,
                    "company": company,
                    "location": location,
                    "link": link,
                    "description": description
                })

for start in range(0, LINKEDIN_PAGES * 25, 25):
    html = get_page(LINKEDIN_URL, params={
        "keywords": "data analyst",
        "location": "Ukraine",
        "start": start
    })

    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("li")

    for card in cards:
        title = get_text(card.select_one(".base-search-card__title"))
        company = get_text(card.select_one(".base-search-card__subtitle"))
        location = get_text(card.select_one(".job-search-card__location"))

        link_tag = card.select_one("a.base-card__full-link") or card.find("a", href=True)

        if link_tag is None:
            continue

        link = link_tag.get("href", "").split("?")[0]

        if link in used_links:
            continue

        used_links.add(link)
        job_id = get_linkedin_job_id(link)

        if job_id:
            details_soup = BeautifulSoup(
                get_page(f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"),
                "html.parser"
            )
            description = get_text(
                details_soup.select_one(".show-more-less-html__markup")
                or details_soup.select_one(".description__text")
                or details_soup.find("body")
            )
        else:
            description = ""

        location = normalize_location(location)

        if JOB_PATTERN.search(title + " " + description):
            jobs.append({
                "platform": "LinkedIn",
                "title": title,
                "company": company,
                "location": location,
                "link": link,
                "description": description
            })

for page in range(1, DJINNI_PAGES + 1):
    html = get_page(add_page(DJINNI_URL, page))
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=re.compile(r"^/jobs/\d"))

    for link_tag in links:
        link = urljoin("https://djinni.co", link_tag.get("href", ""))

        if link in used_links:
            continue

        used_links.add(link)
        card_text = get_text(link_tag.find_parent(["li", "div", "article"]))
        detail_soup = BeautifulSoup(get_page(link), "html.parser")

        title = get_text(detail_soup.find("h1"))
        company = get_text(detail_soup.select_one(".text-secondary.fw-medium"))

        if not company:
            company = "Unknown"

        description = get_text(detail_soup.select_one(".mb-4.job-post__description"))
        location = get_text(detail_soup.select_one(".location-text"))

        if not location:
            location = "Не вказано"

        location = normalize_location(location)

        if JOB_PATTERN.search(title + " " + description):
            jobs.append({
                "platform": "Djinni",
                "title": title,
                "company": company,
                "location": location,
                "link": link,
                "description": description
            })

jobs_df = pd.DataFrame(jobs, columns=["platform", "title", "company", "location", "link", "description"])
jobs_df = jobs_df.drop_duplicates(subset=["platform", "title", "company", "link"])
jobs_df["company"] = jobs_df["company"].fillna("").replace("", "Unknown")
jobs_df["company_key"] = jobs_df["company"].apply(company_key)

reviews = []

companies_df = jobs_df[jobs_df["company"] != "Unknown"][["company", "company_key"]]
companies_df = companies_df.drop_duplicates(subset=["company_key"])

for _, row in companies_df.iterrows():
    company = row["company"]
    key = row["company_key"]
    url = f"https://jobs.dou.ua/companies/{key}/reviews/"
    html = get_page(url)

    if not html:
        continue

    soup = BeautifulSoup(html, "html.parser")
    comments_list = soup.find("div", id="commentsList")

    if comments_list is None:
        continue

    review_count = 0
    comment_blocks = comments_list.find_all("div", class_="b-comment")

    for comment_block in comment_blocks:
        classes = comment_block.get("class", [])

        if any(class_name.startswith("level-") for class_name in classes):
            continue

        comment_div = None

        for child in comment_block.find_all("div", recursive=False):
            classes = child.get("class", [])

            if "comment" in classes:
                comment_div = child
                break

        if comment_div is None:
            continue

        avatar = comment_div.select_one(".avatar")
        prof = comment_div.select_one(".prof")
        date_tag = comment_div.select_one(".comment-link")
        text_tag = comment_div.select_one(".l-text.b-typo")

        if text_tag is None:
            continue

        author = get_text(avatar)
        position = get_text(prof)
        position = re.sub(r"\s+в\s+.+$", "", position)
        review_date = get_text(date_tag)
        review_text = get_text(text_tag)

        if len(review_text) < 40:
            continue

        reviews.append({
            "company": company,
            "company_key": key,
            "review_author": author,
            "review_author_position": clean_text(position),
            "review_date": review_date,
            "review_text": review_text,
            "link": url
        })

        review_count += 1

        if review_count >= DOU_REVIEWS_LIMIT_PER_COMPANY:
            break

reviews_df = pd.DataFrame(reviews, columns=[
    "company",
    "company_key",
    "link",
    "review_date",
    "review_author",
    "review_author_position",
    "review_text",
])
reviews_df = reviews_df.drop_duplicates(subset=["company", "review_text"])

review_text_en_list = []

for review_text in reviews_df["review_text"]:
    review_text_en = translate_text(review_text)
    review_text_en_list.append(review_text_en)

reviews_df["review_text_en"] = review_text_en_list

description_en_list = []

for description in jobs_df["description"]:
    description_en = translate_text(description)
    description_en_list.append(description_en)

jobs_df["description_en"] = description_en_list

jobs_df = jobs_df[[
    "platform",
    "title",
    "company",
    "company_key",
    "location",
    "link",
    "description",
    "description_en"
]]

jobs_df.to_csv(RESULTS_DIR / "jobs_all.csv", index=False, encoding="utf-8")
reviews_df.to_csv(RESULTS_DIR / "company_reviews.csv", index=False, encoding="utf-8")

print("LinkedIn jobs:", len(jobs_df[jobs_df["platform"] == "LinkedIn"]))
print("Work.ua jobs:", len(jobs_df[jobs_df["platform"] == "Work.ua"]))
print("Djinni jobs:", len(jobs_df[jobs_df["platform"] == "Djinni"]))
print("All jobs:", len(jobs_df))
print("Reviews:", len(reviews_df))
print("Files saved to results folder")
