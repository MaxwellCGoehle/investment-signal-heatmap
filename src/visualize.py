import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os

def build_heatmap(ax_heat, ax_trend, pivot_sent, pivot_count):
    def trend_arrow(row):
        vals = row.dropna()
        if len(vals) < 2:
            return "→"
        diff = vals.iloc[-1] - vals.iloc[-2]
        if diff > 0.05:
            return "↑"
        elif diff < -0.05:
            return "↓"
        else:
            return "→"

    trends = pivot_sent.apply(trend_arrow, axis=1)
    trend_colors = {"↑": "#2d6a2d", "↓": "#8b0000", "→": "#555555"}

    annot = pd.DataFrame(index=pivot_sent.index, columns=pivot_sent.columns)
    for sector in pivot_sent.index:
        for week in pivot_sent.columns:
            s = pivot_sent.loc[sector, week]
            c = pivot_count.loc[sector, week]
            if pd.notna(s):
                annot.loc[sector, week] = f"{s:.2f}\n({int(c)})"
            else:
                annot.loc[sector, week] = ""

    sns.heatmap(
        pivot_sent,
        ax=ax_heat,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"label": "Avg Sentiment Score", "shrink": 0.8, "pad": 0.02}
    )

    ax_heat.set_title(
        "Investment Sentiment Heatmap — Alternative Data Signal\n"
        "Source: News NLP (VADER) | Weekly Aggregation | Annotations: Sentiment / Article Count",
        fontsize=12, pad=12
    )
    ax_heat.set_xlabel("Week Starting", fontsize=10)
    ax_heat.set_ylabel("Sector", fontsize=10)
    plt.setp(ax_heat.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.setp(ax_heat.get_yticklabels(), rotation=0, fontsize=9)

    ax_trend.set_xlim(0, 1)
    ax_trend.set_ylim(0, len(trends))
    ax_trend.axis("off")
    ax_trend.set_title("WoW\nTrend", fontsize=10, pad=12)

    for i, (sector, arrow) in enumerate(trends.items()):
        y = len(trends) - i - 0.5
        ax_trend.text(
            0.5, y, arrow,
            ha="center", va="center",
            fontsize=16,
            color=trend_colors[arrow],
            fontweight="bold"
        )


def build_volume_chart(ax, pivot_count):
    sectors = pivot_count.index.tolist()
    weeks = pivot_count.columns.tolist()

    x = np.arange(len(weeks))
    n = len(sectors)
    width = 0.15
    colors = sns.color_palette("Set2", n)

    for i, sector in enumerate(sectors):
        vals = pivot_count.loc[sector].values.astype(float)
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=sector, color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(weeks, rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Week Starting", fontsize=10)
    ax.set_ylabel("Article Count", fontsize=10)
    ax.set_title(
        "News Volume by Sector — Weekly Article Count\n"
        "Indicator of data confidence and media attention per signal",
        fontsize=12, pad=12
    )
    ax.legend(title="Sector", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def build_all(scored):
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

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[11, 1], hspace=0.5)

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_trend = fig.add_subplot(gs[0, 1])
    ax_vol = fig.add_subplot(gs[1, :])

    build_heatmap(ax_heat, ax_trend, pivot_sent, pivot_count)
    build_volume_chart(ax_vol, pivot_count)

    fig.suptitle(
        "Alternative Data Investment Dashboard",
        fontsize=15, fontweight="bold", y=1.01
    )

    return fig


if __name__ == "__main__":
    scored = pd.read_csv("data/scored_articles.csv")

    fig = build_all(scored)
    plt.savefig("visuals/dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

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

    fig1, (ax_heat, ax_trend) = plt.subplots(1, 2, figsize=(16, 5),
                                              gridspec_kw={"width_ratios": [11, 1]})
    build_heatmap(ax_heat, ax_trend, pivot_sent, pivot_count)
    plt.tight_layout()
    plt.savefig("visuals/heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig2, ax_vol = plt.subplots(figsize=(14, 5))
    build_volume_chart(ax_vol, pivot_count)
    plt.tight_layout()
    plt.savefig("visuals/volume.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved dashboard.png, heatmap.png, volume.png")