import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

SECTOR_MAP = {
    "industrial AI manufacturing": "Industrial AI",
    "industrial AI automation": "Industrial AI",
    "cross-border supply chain": "Supply Chain",
    "supply chain disruption trade": "Supply Chain",
    "private equity deal flow": "Private Equity",
    "private equity investment 2026": "Private Equity",
    "venture capital industrial tech": "Venture Capital",
    "venture capital AI funding": "Venture Capital",
    "AI logistics automation": "AI Logistics",
    "AI logistics supply chain": "AI Logistics",
}

def score_articles(df):
    df = df.copy()
    df["sector"] = df["query"].map(SECTOR_MAP)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
    df["week"] = df["publishedAt"].dt.to_period("W").astype(str)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["sentiment"] = df["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    df["article_count"] = 1
    return df

if __name__ == "__main__":
    raw = pd.read_csv("data/raw_articles.csv")
    scored = score_articles(raw)
    scored.to_csv("data/scored_articles.csv", index=False)
    print(scored[["sector", "week", "sentiment"]].head(10))
    print(f"\nSentiment range: {scored['sentiment'].min():.2f} to {scored['sentiment'].max():.2f}")