from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


POSITIVE_LIMIT = 0.7
NEGATIVE_LIMIT = 0.6

RESULTS_DIR = Path("results")
analyzer = SentimentIntensityAnalyzer()

jobs_df = pd.read_csv(RESULTS_DIR / "jobs_all.csv", encoding="utf-8")
reviews_df = pd.read_csv(RESULTS_DIR / "company_reviews.csv", encoding="utf-8")

job_scores = []
job_sentiments = []

for description in jobs_df["description_en"].fillna(""):
    vader_score = analyzer.polarity_scores(str(description))["compound"]

    if vader_score > POSITIVE_LIMIT:
        label = "positive"
    elif vader_score < NEGATIVE_LIMIT:
        label = "negative"
    else:
        label = "neutral"

    job_scores.append(round(vader_score, 4))
    job_sentiments.append(label)

jobs_df["job_tonality_score"] = job_scores
jobs_df["job_sentiment"] = job_sentiments

review_scores = []
review_sentiments = []

for review_text in reviews_df["review_text_en"].fillna(""):
    vader_score = analyzer.polarity_scores(str(review_text))["compound"]

    if vader_score > POSITIVE_LIMIT:
        label = "positive"
    elif vader_score < NEGATIVE_LIMIT:
        label = "negative"
    else:
        label = "neutral"

    review_scores.append(round(vader_score, 4))
    review_sentiments.append(label)

reviews_df["review_tonality_score"] = review_scores
reviews_df["review_sentiment"] = review_sentiments

jobs_summary = jobs_df.groupby("company_key").agg(
    company=("company", "first"),
    vacancies=("title", "count"),
    platforms=("platform", lambda values: ", ".join(sorted(set(values)))),
    avg_job_tonality_score=("job_tonality_score", "mean")
).reset_index()

company_reviews = reviews_df.groupby("company_key").agg(
    review_company=("company", "first"),
    reviews_count=("review_text", "count"),
    avg_review_tonality_score=("review_tonality_score", "mean"),
    positive_reviews=("review_sentiment", lambda values: (values == "positive").sum()),
    neutral_reviews=("review_sentiment", lambda values: (values == "neutral").sum()),
    negative_reviews=("review_sentiment", lambda values: (values == "negative").sum())
).reset_index()

final_summary = jobs_summary.merge(company_reviews, on="company_key", how="outer")
final_summary["company"] = final_summary["company"].fillna(final_summary["review_company"])
final_summary = final_summary.drop(columns=["company_key", "review_company"])

final_summary["vacancies"] = final_summary["vacancies"].fillna(0).astype(int)
final_summary["reviews_count"] = final_summary["reviews_count"].fillna(0).astype(int)
final_summary["positive_reviews"] = final_summary["positive_reviews"].fillna(0).astype(int)
final_summary["neutral_reviews"] = final_summary["neutral_reviews"].fillna(0).astype(int)
final_summary["negative_reviews"] = final_summary["negative_reviews"].fillna(0).astype(int)

final_summary = final_summary.sort_values(by=["vacancies", "reviews_count"], ascending=False)

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

jobs_df.to_csv(RESULTS_DIR / "jobs_all_with_sentiment.csv", index=False, encoding="utf-8")
reviews_df.to_csv(RESULTS_DIR / "company_reviews_with_sentiment.csv", index=False, encoding="utf-8")
final_summary.to_csv(RESULTS_DIR / "company_tonality_summary.csv", index=False, encoding="utf-8")

print("Jobs with sentiment:", len(jobs_df))
print("Reviews with sentiment:", len(reviews_df))
print("Files saved to results folder")
