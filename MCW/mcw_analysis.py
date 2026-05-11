import os
import re
import math
from collections import Counter, defaultdict

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
import networkx as nx
from bs4 import BeautifulSoup


RESULTS_DIR = "results"
RAW_NEWS_CSV = os.path.join(RESULTS_DIR, "news_raw.csv")
MODEL_FILE = os.path.join(RESULTS_DIR, "news_category_model.joblib")
VECTORIZER_FILE = os.path.join(RESULTS_DIR, "news_category_vectorizer.joblib")
CATEGORIES = ["politics", "security", "economy"]

os.makedirs(RESULTS_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")


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


def get_tokens(text):
    tokens = []

    for token in str(text).split():
        token = token.lower().strip()
        if token.isalpha() and len(token) > 2:
            tokens.append(token)

    return tokens


def predict_category(text, model, vectorizer):
    model_text = clean_dataset_text(text)

    if len(model_text.split()) < 2:
        return "politics"

    x_text = vectorizer.transform([model_text])
    return model.predict(x_text)[0]


def plot_category_distribution(df):
    counts = df["category"].value_counts().reindex(CATEGORIES).fillna(0)
    shares = counts / counts.sum()

    pd.DataFrame({
        "category": counts.index,
        "count": counts.values,
        "share": shares.values
    }).to_csv(os.path.join(RESULTS_DIR, "category_distribution.csv"), index=False, encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("News category distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of news")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "category_distribution.png"), dpi=200)
    plt.close()


def plot_source_category_distribution(df):
    table = pd.crosstab(df["source"], df["category"])
    table = table.reindex(columns=CATEGORIES).fillna(0)
    table.to_csv(os.path.join(RESULTS_DIR, "source_category_counts.csv"), encoding="utf-8")

    table.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("News categories by source")
    plt.xlabel("Source")
    plt.ylabel("Number of news")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "source_category_distribution.png"), dpi=200)
    plt.close()


def compute_category_statistics(docs):
    document_count = len(docs)
    total_frequency = Counter()
    tf_sum = defaultdict(float)
    document_frequency = Counter()

    for tokens in docs:
        total_frequency.update(tokens)
        token_count = len(tokens)
        counts = Counter(tokens)

        for term, count in counts.items():
            if token_count > 0:
                tf_sum[term] += count / token_count
            document_frequency[term] += 1

    average_tf = {}
    idf = {}

    for term in total_frequency.keys():
        average_tf[term] = tf_sum[term] / document_count
        idf[term] = math.log(document_count / document_frequency[term])

    return average_tf, idf, total_frequency, document_frequency


def save_top_table_and_plot(items, csv_path, png_path, title, x_label, y_label):
    table = pd.DataFrame(items, columns=[x_label.lower().replace(" ", "_"), y_label.lower().replace(" ", "_")])
    table.to_csv(csv_path, index=False, encoding="utf-8")

    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def plot_lexical_dispersion(docs, top_terms, output_path, category):
    x_values = []
    y_values = []

    for tokens in docs:
        if len(tokens) < 2:
            continue

        for i, token in enumerate(tokens):
            if token in top_terms:
                x_values.append(i / (len(tokens) - 1))
                y_values.append(top_terms.index(token))

    plt.figure(figsize=(12, 5))
    plt.scatter(x_values, y_values, s=16, alpha=0.6)
    plt.yticks(range(len(top_terms)), top_terms)
    plt.xlim(0, 1)
    plt.title("Lexical dispersion: " + category)
    plt.xlabel("Normalized position in news")
    plt.ylabel("Top TF words")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_word_length_distribution(df):
    plt.figure(figsize=(10, 6))
    rows = []

    for category in CATEGORIES:
        text = " ".join(df[df["category"] == category]["analysis_text"].fillna(""))
        tokens = get_tokens(text)
        lengths = [len(token) for token in tokens]
        counts = Counter(lengths)
        max_length = min(max(counts.keys()), 20)
        x_values = list(range(1, max_length + 1))
        y_values = [counts.get(length, 0) for length in x_values]

        for length, count in zip(x_values, y_values):
            rows.append({
                "category": category,
                "word_length": length,
                "count": count
            })

        plt.plot(x_values, y_values, marker="o", label=category)

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "word_length_distribution.csv"), index=False, encoding="utf-8")

    plt.title("Word length distribution by category")
    plt.xlabel("Word length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "word_length_distribution.png"), dpi=200)
    plt.close()


def make_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def plot_ngram_analysis(category_docs):
    plt.figure(figsize=(10, 6))
    rows = []

    for category in CATEGORIES:
        y_values = []

        for n in range(1, 6):
            counter = Counter()

            for tokens in category_docs[category]:
                counter.update(make_ngrams(tokens, n))

            repeated_count = len([item for item in counter.values() if item > 1])
            y_values.append(repeated_count)
            rows.append({
                "category": category,
                "n": n,
                "repeated_ngrams": repeated_count
            })

        plt.plot(range(1, 6), y_values, marker="o", label=category)

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "ngram_repetition.csv"), index=False, encoding="utf-8")

    plt.title("N-gram repetition by category")
    plt.xlabel("N-gram size")
    plt.ylabel("Repeated n-grams count")
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ngram_repetition.png"), dpi=200)
    plt.close()


def plot_bigram_graph(docs, category_dir, category):
    counter = Counter()

    for tokens in docs:
        counter.update(make_ngrams(tokens, 2))

    bigrams = [((first, second), count) for (first, second), count in counter.most_common() if count >= 3]

    pd.DataFrame(
        [(first + " " + second, count) for (first, second), count in bigrams],
        columns=["bigram", "count"]
    ).to_csv(os.path.join(category_dir, "bigrams.csv"), index=False, encoding="utf-8")

    graph = nx.DiGraph()

    for (first, second), count in bigrams:
        graph.add_edge(first, second, weight=count)

    plt.figure(figsize=(16, 10))
    position = nx.spring_layout(graph, seed=42, k=1.6, iterations=100)

    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    max_weight = max(weights)
    widths = [2 + 8 * weight / max_weight for weight in weights]

    nx.draw_networkx_edges(
        graph,
        position,
        width=widths,
        edge_color="black",
        alpha=0.75,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=18,
        min_target_margin=18
    )

    nx.draw_networkx_nodes(
        graph,
        position,
        node_size=1200,
        node_color="white",
        edgecolors="black",
        linewidths=1.5,
        alpha=0.95
    )

    nx.draw_networkx_labels(graph, position, font_size=9, font_weight="bold")

    edge_labels = {}
    for first, second in graph.edges():
        edge_labels[(first, second)] = graph[first][second]["weight"]

    nx.draw_networkx_edge_labels(
        graph,
        position,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.55,
        rotate=False
    )

    plt.title("Bigram graph: " + category)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(category_dir, "bigram_graph.png"), dpi=250)
    plt.close()


model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

news_df = pd.read_csv(RAW_NEWS_CSV)
news_df["translated_text"] = news_df["translated_text"].fillna(news_df["full_text"].fillna(""))
news_df["analysis_text"] = news_df["translated_text"].apply(clean_dataset_text)
news_df["category"] = news_df["translated_text"].apply(lambda text: predict_category(text, model, vectorizer))

news_df.to_csv(os.path.join(RESULTS_DIR, "news_analysis_result.csv"), index=False, encoding="utf-8")

plot_category_distribution(news_df)
plot_source_category_distribution(news_df)
plot_word_length_distribution(news_df)

category_docs = {}

for category in CATEGORIES:
    category_dir = os.path.join(RESULTS_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    category_news = news_df[news_df["category"] == category]
    docs = [get_tokens(text) for text in category_news["analysis_text"].fillna("")]
    category_docs[category] = docs

    average_tf, idf, total_frequency, document_frequency = compute_category_statistics(docs)

    tf_idf = {}
    for term in average_tf.keys():
        tf_idf[term] = average_tf[term] * idf[term]

    top_tf = sorted(average_tf.items(), key=lambda item: item[1], reverse=True)[:10]
    top_idf = sorted(idf.items(), key=lambda item: item[1])[:10]
    top_tfidf = sorted(tf_idf.items(), key=lambda item: item[1], reverse=True)[:10]
    top_frequency = total_frequency.most_common(10)

    save_top_table_and_plot(
        top_tf,
        os.path.join(category_dir, "tf_top10.csv"),
        os.path.join(category_dir, "tf_top10.png"),
        "TF Top 10: " + category,
        "Term",
        "Average TF"
    )

    save_top_table_and_plot(
        top_idf,
        os.path.join(category_dir, "idf_top10.csv"),
        os.path.join(category_dir, "idf_top10.png"),
        "IDF Top 10: " + category,
        "Term",
        "IDF"
    )

    save_top_table_and_plot(
        top_tfidf,
        os.path.join(category_dir, "tfidf_top10.csv"),
        os.path.join(category_dir, "tfidf_top10.png"),
        "TF-IDF Top 10: " + category,
        "Term",
        "TF-IDF"
    )

    save_top_table_and_plot(
        top_frequency,
        os.path.join(category_dir, "frequency_top10.csv"),
        os.path.join(category_dir, "frequency_top10.png"),
        "Frequency Top 10: " + category,
        "Term",
        "Frequency"
    )

    rows = []
    for term in sorted(total_frequency.keys()):
        rows.append({
            "term": term,
            "average_tf": average_tf[term],
            "idf": idf[term],
            "tf_idf": tf_idf[term],
            "frequency": total_frequency[term],
            "document_frequency": document_frequency[term]
        })

    pd.DataFrame(rows).to_csv(os.path.join(category_dir, "category_term_statistics.csv"), index=False, encoding="utf-8")

    top_tf_terms = [term for term, value in top_tf]
    plot_lexical_dispersion(
        docs,
        top_tf_terms,
        os.path.join(category_dir, "lexical_dispersion.png"),
        category
    )

    plot_bigram_graph(docs, category_dir, category)

plot_ngram_analysis(category_docs)

print()
print("Analysis completed")
print("News analyzed:", len(news_df))
print("Results saved to:", RESULTS_DIR)
print(news_df[["source", "feed_topic", "category", "title"]].head(30))
