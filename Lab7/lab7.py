import os
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin
from collections import Counter

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from gtts import gTTS
import speech_recognition as sr

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


URL = "https://www.ukr.net/news/main.html"
CSV_FILE = "news_ukrnet.csv"
TRAIN_FILE = "news_train_text.txt"
MODEL_FILE = "news_lstm_from_parsing.pt"
AUDIO_DIR = "bot_audio"

MAX_NEWS = 90
SEQ_LEN = 180
BATCH_SIZE = 12
EPOCHS = 140
HIDDEN_SIZE = 192
EMBEDDING_SIZE = 128
LEARNING_RATE = 0.003

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

EXTRA_STOP_WORDS = {
    "новина", "новини", "стрічка", "сьогодні", "зараз",
    "останні", "остання", "головні", "головна",
    "свіжі", "актуальні", "дай", "покажи", "розкажи",
    "мені", "будь", "ласка", "інфопростір", "інфопросторі"
}

nlp = spacy.load("uk_core_news_sm")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=2):
        super(TextRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=0.25)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size=1):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return h, c


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'").replace("ʼ", "'")
    text = re.sub(r"[^а-щьюяґєіїa-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text):
    text = normalize_text(text)
    doc = nlp(text)
    words = []

    for token in doc:
        lemma = token.lemma_.lower().strip()

        if len(lemma) >= 3 and token.is_alpha and not token.is_stop and lemma not in EXTRA_STOP_WORDS:
            words.append(lemma)

    return " ".join(words)


def parse_news():
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "lxml")
    container = soup.select_one("article")

    parsed_news = []
    current_date = datetime.now().date()
    previous_minutes = None

    for section_tag in container.find_all("section"):
        time_tag = section_tag.find("time")
        link_tag = section_tag.find("a")

        if time_tag is not None and link_tag is not None:
            time_text = clean_text(time_tag.get_text())
            title = clean_text(link_tag.get_text(" "))
            href = link_tag.get("href", "").strip()
            link = urljoin(URL, href)

            hours, minutes = map(int, time_text.split(":"))
            current_minutes = hours * 60 + minutes

            if previous_minutes is not None and current_minutes > previous_minutes:
                current_date -= timedelta(days=1)

            date_text = f"{current_date.day} {MONTHS[current_date.month]}"
            titles = [item["title"] for item in parsed_news]

            if title not in titles:
                parsed_news.append({
                    "date": date_text,
                    "time": time_text,
                    "title": title,
                    "link": link,
                    "search_text": lemmatize_text(title),
                })

            previous_minutes = current_minutes

    parsed_news = parsed_news[:MAX_NEWS]
    pd.DataFrame(parsed_news).to_csv(CSV_FILE, index=False, encoding="utf-8")
    print("Знайдено новин:", len(parsed_news))

    return parsed_news


def make_training_text(news):
    text = ""
    text += "Поточна ситуація в інфопросторі за результатами парсингу сайту ukr.net.\n"
    text += "Останні новини у стрічці:\n"

    for i, item in enumerate(news, 1):
        text += f"{i}. Дата: {item['date']}. Час: {item['time']}. Заголовок: {item['title']}.\n"

    text += "\nНовини за датами:\n"
    dates = sorted(set([item["date"] for item in news]))

    for date_text in dates:
        text += f"Новини за {date_text}:\n"

        for item in news:
            if item["date"] == date_text:
                text += f"- {item['time']}. {item['title']}.\n"

    text += "\nТематичні фрагменти стрічки:\n"

    for item in news:
        doc = nlp(item["title"])
        keywords = []

        for token in doc:
            lemma = token.lemma_.lower().strip()

            if len(lemma) >= 4 and token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and lemma not in EXTRA_STOP_WORDS:
                keywords.append(lemma)

        if len(keywords) > 0:
            text += "Тема: " + ", ".join(keywords[:5]) + ". "
            text += f"Новина: {item['date']}, {item['time']}. {item['title']}.\n"

    with open(TRAIN_FILE, "w", encoding="utf-8") as file:
        file.write(text)

    return text


def make_char_dataset(text):
    char_counts = Counter(text)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_chars = [char for char, count in char_counts]

    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text])

    return sequence, char_to_idx, idx_to_char


def train_lstm(sequence, char_to_idx, idx_to_char):
    model = TextRNN(
        input_size=len(idx_to_char),
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE,
        n_layers=2
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    print("Training LSTM text generator...")

    best_loss = 1000000
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        trains = []
        targets = []

        for i in range(BATCH_SIZE):
            batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
            chunk = sequence[batch_start: batch_start + SEQ_LEN]
            train_part = torch.LongTensor(chunk[:-1]).view(-1, 1)
            target_part = torch.LongTensor(chunk[1:]).view(-1, 1)
            trains.append(train_part)
            targets.append(target_part)

        train = torch.stack(trains, dim=0)
        target = torch.stack(targets, dim=0)

        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)

        hidden = model.init_hidden(BATCH_SIZE)
        output, hidden = model(train, hidden)

        loss = criterion(
            output.permute(1, 2, 0),
            target.squeeze(-1).permute(1, 0)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", round(loss.item(), 4))

        if epoch >= 70 and bad_epochs >= 30:
            print("Early stopping at epoch:", epoch)
            break

    torch.save({
        "model_state": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }, MODEL_FILE)

    return model


def analyze_question(question, news):
    q_norm = normalize_text(question)

    date_result = re.search(r"(\d{1,2})\s+(січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)", q_norm)
    question_date = None

    if date_result is not None:
        question_date = f"{int(date_result.group(1))} {date_result.group(2)}"

    if "сьогодні" in q_norm and len(news) > 0:
        question_date = news[0]["date"]

    text_without_date = re.sub(r"\d{1,2}\s+(січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)", " ", q_norm)
    text_without_date = clean_text(text_without_date)

    topic = ""
    has_topic_marker = False
    patterns = [
        r"про\s+(.+)",
        r"щодо\s+(.+)",
        r"стосовно\s+(.+)",
        r"по\s+тематиці\s+(.+)",
        r"по\s+темі\s+(.+)",
        r"на\s+тему\s+(.+)"
    ]

    for pattern in patterns:
        topic_result = re.search(pattern, text_without_date)

        if topic_result is not None:
            topic = topic_result.group(1)
            has_topic_marker = True
            break

    news_words = ["новина", "новини", "новин", "останні", "свіжі", "актуальні", "головні", "стрічка", "інфопростір", "інфопросторі"]
    has_news_words = any(word in q_norm for word in news_words)
    has_count_words = "скільки" in q_norm or "кількість" in q_norm

    if not has_news_words and question_date is None and not has_topic_marker and not has_count_words:
        return q_norm, None, "", False

    if topic == "" and has_topic_marker:
        topic = text_without_date

    if topic == "" and not has_news_words and question_date is None:
        return q_norm, None, "", False

    topic = lemmatize_text(topic)

    return q_norm, question_date, topic, True


def select_news(news, question_date, topic):
    selected_news = news

    if question_date is not None:
        selected_news = [item for item in news if item["date"] == question_date]

    if topic != "":
        if topic in ["україна"] or topic.startswith("україн"):
            return selected_news[:5]

        topic_words = set(topic.split())
        exact_news = []

        for item in selected_news:
            title_words = set(item["search_text"].split())

            if len(topic_words.intersection(title_words)) > 0:
                exact_news.append(item)

        if len(exact_news) > 0:
            return exact_news[:5]

        current_titles = [item["search_text"] for item in selected_news]

        if len(current_titles) == 0:
            return []

        local_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), lowercase=True)
        local_vectors = local_vectorizer.fit_transform(current_titles)
        question_vector = local_vectorizer.transform([topic])
        similarities = cosine_similarity(question_vector, local_vectors)[0]

        topic_news = []

        for index in range(len(selected_news)):
            if similarities[index] >= 0.18:
                topic_news.append(selected_news[index])

            if len(topic_news) >= 5:
                break

        selected_news = topic_news

    return selected_news


def make_seed(question, news):
    q_norm, question_date, topic, is_news_question = analyze_question(question, news)

    if not is_news_question:
        return "Бот відповідає тільки на питання про поточні новини, дату новин або тему новин. "

    if "скільки" in q_norm or "кількість" in q_norm:
        return f"Було зібрано {len(news)} новин із сайту ukr.net. "

    selected_news = select_news(news, question_date, topic)

    if len(selected_news) == 0:
        if question_date is not None:
            return f"У стрічці немає новин за {question_date}. "
        return "У стрічці новин немає точного збігу. "

    if question_date is not None and topic != "":
        seed = f"Новини за {question_date} за тематикою {topic}: "
    elif question_date is not None:
        seed = f"Новини за {question_date}: "
    elif topic != "":
        seed = f"Останні новини за тематикою {topic}: "
    else:
        seed = "Останні новини у стрічці: "

    for i, item in enumerate(selected_news[:5], 1):
        seed += f"{i}. {item['date']}, {item['time']}. {item['title']}. "

    return seed


def generate_answer(question, news, model, char_to_idx, idx_to_char):
    q_norm, question_date, topic, is_news_question = analyze_question(question, news)

    if not is_news_question:
        return "Бот відповідає тільки на питання про поточні новини, дату новин або тему новин."

    seed = make_seed(question, news)
    model.eval()

    clean_seed = ""

    for char in seed:
        if char in char_to_idx:
            clean_seed += char

    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in clean_seed]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)

    answer = clean_seed

    with torch.no_grad():
        output, hidden = model(train, hidden)
        inp = train[-1].view(-1, 1, 1)

        for i in range(35):
            output, hidden = model(inp.to(device), hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = F.softmax(output_logits / 0.08, dim=-1).detach().cpu().data.numpy()
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            predicted_char = idx_to_char[top_index]
            answer += predicted_char
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)

            if answer.count(".") >= clean_seed.count(".") + 1:
                break

    answer = clean_text(answer)
    tail = answer[len(clean_seed):].strip()
    bad_tail = False

    if len(tail) > 0:
        if not tail.endswith((".", "!", "?")):
            bad_tail = True

        if "Дата:" in tail or "Заголовок:" in tail or "Тема:" in tail or "Новина:" in tail or "http" in tail:
            bad_tail = True

        if tail.count(":") > 0:
            bad_tail = True

        tail_words = tail.lower().split()

        for word in tail_words:
            if len(word) < 2:
                bad_tail = True

        for i in range(len(tail_words) - 1):
            if tail_words[i] == tail_words[i + 1]:
                bad_tail = True

    if bad_tail:
        answer = seed

    if len(answer) > len(seed) + 120:
        answer = seed

    return clean_text(answer)


def voice_question():
    recognizer = sr.Recognizer()
    typed_question = input("\n[ENTER] Почати голосове введення або введіть питання текстом: ").strip()

    if typed_question != "":
        return typed_question

    try:
        with sr.Microphone() as source:
            print("Підготовка мікрофона...")
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 1.8
            recognizer.non_speaking_duration = 1.0
            recognizer.phrase_threshold = 0.15
            print("Говоріть зараз...")
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=18)

        print("Розпізнавання мовлення...")
        question = recognizer.recognize_google(audio, language="uk-UA")
        question = clean_text(question)
        print("Питання:", question)

        q_check = normalize_text(question)

        if q_check != "" and q_check.split()[-1] in ["про", "щодо", "стосовно", "тему", "темі", "тематиці", "за"]:
            print("Скажіть пропущене слово зараз")

            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=8)

            addition = recognizer.recognize_google(audio, language="uk-UA")
            question = clean_text(question + " " + addition)
            print("Повне питання:", question)

    except:
        print("Помилка голосового введення")
        question = input("Введіть питання текстом: ").strip()

    return question


def speak_answer(text, number):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_path = os.path.join(AUDIO_DIR, f"answer_{number}.mp3")
    gTTS(text=text, lang="uk", slow=False).save(audio_path)
    print("Аудіовідповідь збережено:", audio_path)
    os.startfile(audio_path)


news = parse_news()
training_text = make_training_text(news)
sequence, char_to_idx, idx_to_char = make_char_dataset(training_text)
model = train_lstm(sequence, char_to_idx, idx_to_char)

print("\nГолосовий новинний бот готовий до роботи.")
print("Приклади питань:")
print("останні новини")
print("останні новини про Дніпро")
print("останні новини про Україну")
print("новини за 5 травня")
print("новини за 5 травня про Дніпро")
print("Для завершення введіть або скажіть: стоп")

question_number = 1

while True:
    question = voice_question()
    q_norm = normalize_text(question)

    if q_norm in ["стоп", "stop", "exit", "вихід"]:
        final_text = "Роботу бота завершено."
        print(final_text)
        speak_answer(final_text, question_number)
        break

    answer = generate_answer(question, news, model, char_to_idx, idx_to_char)

    print("\nВідповідь:")
    print(answer)

    speak_answer(answer, question_number)

    question_number += 1
