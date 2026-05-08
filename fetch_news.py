import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NewsAPI")

QUERIES = [
    "industrial AI manufacturing",
    "cross-border supply chain",
    "private equity deal flow",
    "venture capital industrial tech",
    "AI logistics automation",
]

def fetch_articles(query, days_back=30):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 50,
        "apiKey": API_KEY,
    }
    r = requests.get(url, params=params)
    articles = r.json().get("articles", [])
    df = pd.DataFrame(articles)
    df["query"] = query
    return df

if __name__ == "__main__":
    all_articles = pd.concat([fetch_articles(q) for q in QUERIES], ignore_index=True)
    all_articles.to_csv("/Users/maxwellgoehle/investment-signal-heatmap/raw_articles.csv", index=False)
    print(f"Fetched {len(all_articles)} articles")
    print(all_articles[["publishedAt", "title", "query"]].head(10))