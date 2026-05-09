import os
import pandas as pd
from src.fetch_news import fetch_articles, QUERIES
from src.score_articles import score_articles
from src.visualize import build_all, build_heatmap, build_volume_chart
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    print(" Alternative Data Investment Dashboard \n")    
    timestamp = datetime.now().strftime("%Y%m%d")

    # Fetch
    print("[1/3] Fetching news articles...")
    os.makedirs("data", exist_ok=True)
    all_articles = pd.concat([fetch_articles(q) for q in QUERIES], ignore_index=True)
    all_articles.to_csv(f"data/raw_{timestamp}.csv", index=False)
    print(f"    Fetched {len(all_articles)} articles\n")

    # Score
    print("[2/3] Scoring sentiment...")
    scored = score_articles(all_articles)
    scored.to_csv(f"data/scored_{timestamp}.csv", index=False)
    print(f"    Scored {len(scored)} articles")
    print(f"    Sentiment range: {scored['sentiment'].min():.2f} to {scored['sentiment'].max():.2f}\n")

    # Visualize
    print("[3/3] Building visualizations...")

    scored["article_count"] = 1
    agg = scored.groupby(["sector", "week"]).agg(
        avg_sentiment=("sentiment", "mean"),
        article_count=("article_count", "sum")
    ).reset_index()

    pivot_sent = agg.pivot(index="sector", columns="week", values="avg_sentiment")
    pivot_count = agg.pivot(index="sector", columns="week", values="article_count")
    pivot_sent = pivot_sent.sort_index(axis=1)
    pivot_count = pivot_count.sort_index(axis=1)
    pivot_sent.columns = [w.split("/")[0] for w in pivot_sent.columns]
    pivot_count.columns = [w.split("/")[0] for w in pivot_count.columns]

    # Dashboard
    fig = build_all(scored)
    plt.savefig(f"visuals/dashboard_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmap
    fig1, (ax_heat, ax_trend) = plt.subplots(1, 2, figsize=(16, 5),
                                              gridspec_kw={"width_ratios": [11, 1]})
    build_heatmap(ax_heat, ax_trend, pivot_sent, pivot_count)
    plt.tight_layout()
    plt.savefig(f"visuals/heatmap_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Volume
    fig2, ax_vol = plt.subplots(figsize=(14, 5))
    build_volume_chart(ax_vol, pivot_count)
    plt.tight_layout()
    plt.savefig(f"visuals/volume_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved: visuals/dashboard_{timestamp}.png")
    print(f"    Saved: visuals/heatmap_{timestamp}.png")
    print(f"    Saved: visuals/volume_{timestamp}.png")
    print("\n Open visuals/ to view results.")

if __name__ == "__main__":
    main()