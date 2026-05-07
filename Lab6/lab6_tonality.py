import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FILE = Path("train.csv")
TEST_FILE = Path("test.csv")
JOBS_FILE = RESULTS_DIR / "jobs_all.csv"
REVIEWS_FILE = RESULTS_DIR / "company_reviews.csv"

TRAIN_ROWS = 100000
TEST_ROWS = 15000
BATCH_SIZE = 32
EPOCHS = 5
MAX_TOKENS = 12000
MAX_LENGTH = 180
RANDOM_STATE = 42

clean_text = lambda text: re.sub(r"\s+", " ", str(text).replace("\xa0", " ").replace("\n", " ")).strip()

print("Reading train.csv and test.csv")

train_df = pd.read_csv(
    TRAIN_FILE,
    header=None,
    names=["label", "text"],
    nrows=TRAIN_ROWS,
    encoding="utf-8"
)

test_df = pd.read_csv(
    TEST_FILE,
    header=None,
    names=["label", "text"],
    nrows=TEST_ROWS,
    encoding="utf-8"
)

train_df["label"] = train_df["label"].replace({1: 0, 2: 1}).astype("float32")
test_df["label"] = test_df["label"].replace({1: 0, 2: 1}).astype("float32")

train_df["text"] = train_df["text"].apply(clean_text)
test_df["text"] = test_df["text"].apply(clean_text)

train_df = train_df[train_df["text"].str.len() > 10].copy()
test_df = test_df[test_df["text"].str.len() > 10].copy()

train_df.to_csv(RESULTS_DIR / "train_prepared.csv", index=False, encoding="utf-8")
test_df.to_csv(RESULTS_DIR / "test_prepared.csv", index=False, encoding="utf-8")

train_texts = train_df["text"].astype(str).values
train_labels = train_df["label"].values

test_texts = test_df["text"].astype(str).values
test_labels = test_df["label"].values

train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
train_dataset = train_dataset.shuffle(10000, seed=RANDOM_STATE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Training examples:", len(train_df))
print("Test examples:", len(test_df))
print(train_df["label"].value_counts().rename(index={0.0: "negative", 1.0: "positive"}))

encoder = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_sequence_length=MAX_LENGTH
)

encoder.adapt(train_texts)

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[early_stop]
)

history_data = history.history

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history_data["accuracy"])
plt.plot(history_data["val_accuracy"])
plt.title("Training and Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Test Accuracy"])

plt.subplot(1, 2, 2)
plt.plot(history_data["loss"])
plt.plot(history_data["val_loss"])
plt.title("Training and Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Test Loss"])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "lstm_training.png", dpi=150)
plt.close()

print("Checking model on test.csv")

test_scores = model.predict(np.array(test_df["text"].tolist(), dtype=object), verbose=0).reshape(-1)

test_result = test_df.copy()
test_result["tonality_score"] = [round(float(score), 4) for score in test_scores]
test_result["predicted_sentiment"] = [
    "negative" if score <= 0.33 else "neutral" if score < 0.66 else "positive"
    for score in test_scores
]
test_result["true_sentiment"] = test_result["label"].map({0.0: "negative", 1.0: "positive"})

test_binary_predictions = [1.0 if score >= 0.5 else 0.0 for score in test_scores]
test_accuracy = accuracy_score(test_df["label"].values, test_binary_predictions)

test_result.to_csv(RESULTS_DIR / "test_with_predictions.csv", index=False, encoding="utf-8")

print("Test accuracy:", round(float(test_accuracy), 4))

print("Reading scraping results")

jobs_df = pd.read_csv(JOBS_FILE, encoding="utf-8")
reviews_df = pd.read_csv(REVIEWS_FILE, encoding="utf-8")

print("Predicting sentiment for vacancies")

job_texts = jobs_df["description_en"].fillna("").apply(clean_text).values
job_scores = model.predict(np.array(job_texts, dtype=object), verbose=0).reshape(-1)

jobs_df["job_tonality_score"] = [round(float(score), 4) for score in job_scores]
jobs_df["job_sentiment"] = [
    "negative" if score <= 0.33 else "neutral" if score < 0.66 else "positive"
    for score in job_scores
]

print("Predicting sentiment for company reviews")

review_texts = reviews_df["review_text_en"].fillna("").apply(clean_text).values
review_scores = model.predict(np.array(review_texts, dtype=object), verbose=0).reshape(-1)

reviews_df["review_tonality_score"] = [round(float(score), 4) for score in review_scores]
reviews_df["review_sentiment"] = [
    "negative" if score <= 0.33 else "neutral" if score < 0.66 else "positive"
    for score in review_scores
]

print("Creating company summary")

jobs_summary = jobs_df.groupby("company_key").agg(
    company=("company", "first"),
    vacancies=("title", "count"),
    platforms=("platform", lambda values: ", ".join(sorted(set(values)))),
    avg_job_tonality_score=("job_tonality_score", "mean")
).reset_index()

reviews_summary = reviews_df.groupby("company_key").agg(
    reviews_count=("review_text", "count"),
    avg_review_tonality_score=("review_tonality_score", "mean"),
    positive_reviews=("review_sentiment", lambda values: (values == "positive").sum()),
    neutral_reviews=("review_sentiment", lambda values: (values == "neutral").sum()),
    negative_reviews=("review_sentiment", lambda values: (values == "negative").sum())
).reset_index()

company_summary = jobs_summary.merge(reviews_summary, on="company_key", how="left")

company_summary["reviews_count"] = company_summary["reviews_count"].fillna(0).astype(int)
company_summary["positive_reviews"] = company_summary["positive_reviews"].fillna(0).astype(int)
company_summary["neutral_reviews"] = company_summary["neutral_reviews"].fillna(0).astype(int)
company_summary["negative_reviews"] = company_summary["negative_reviews"].fillna(0).astype(int)

company_summary = company_summary.drop(columns=["company_key"])
company_summary = company_summary.sort_values(by=["vacancies", "reviews_count"], ascending=False)

jobs_df = jobs_df[[
    "platform",
    "title",
    "company",
    "company_key",
    "location",
    "link",
    "description",
    "description_en",
    "job_tonality_score",
    "job_sentiment"
]]

reviews_df = reviews_df[[
    "company",
    "company_key",
    "link",
    "review_date",
    "review_author",
    "review_author_position",
    "review_text",
    "review_text_en",
    "review_tonality_score",
    "review_sentiment"
]]

jobs_df.to_csv(RESULTS_DIR / "jobs_all_with_sentiment.csv", index=False, encoding="utf-8")
reviews_df.to_csv(RESULTS_DIR / "company_reviews_with_sentiment.csv", index=False, encoding="utf-8")
company_summary.to_csv(RESULTS_DIR / "company_tonality_summary.csv", index=False, encoding="utf-8")

print("Jobs with sentiment:", len(jobs_df))
print("Reviews with sentiment:", len(reviews_df))
print("Summary rows:", len(company_summary))
print("Files saved to results folder")
