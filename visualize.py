import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_heatmap(scored):
    scored["article_count"] = 1
    
    agg = scored.groupby(["sector", "week"]).agg(
        avg_sentiment=("sentiment", "mean"),
        article_count=("article_count", "sum")
    ).reset_index()

    pivot = agg.pivot(index="sector", columns="week", values="avg_sentiment")
    pivot = pivot.sort_index(axis=1)

    # Shorten week labels to just start date
    pivot.columns = [w.split("/")[0] for w in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 5))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "Avg Sentiment Score", "shrink": 0.8}
    )

    ax.set_title(
        "Investment Sentiment Heatmap — Alternative Data Signal\nSource: News NLP (VADER) across target sectors | Weekly Aggregation",
        fontsize=13,
        pad=15
    )
    ax.set_xlabel("Week Starting", fontsize=11)
    ax.set_ylabel("Sector", fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    return fig

if __name__ == "__main__":
    scored = pd.read_csv("scored_articles.csv")
    fig = build_heatmap(scored)
    plt.savefig("visuals/heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved to visuals/heatmap.png")