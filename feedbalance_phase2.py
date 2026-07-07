"""FeedBalance Phase 2: data science analysis and charts."""

import json
import math
import os
import random
import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="FeedBalance Phase 2",
    page_icon="FB",
    layout="wide",
)


CATEGORIES = [
    "fitness", "food", "comedy", "love", "news",
    "motivation", "quotes", "cricket", "tech", "music",
]

CAT_COLOR = {
    "fitness": "#ef4444",
    "food": "#f97316",
    "comedy": "#eab308",
    "love": "#ec4899",
    "news": "#6366f1",
    "motivation": "#f59e0b",
    "quotes": "#8b5cf6",
    "cricket": "#10b981",
    "tech": "#3b82f6",
    "music": "#06b6d4",
}

SAMPLE_TITLES = {
    "fitness": ["5 Best Morning Workouts", "Chest Day Routine", "Lose Weight in 30 Days", "Home Gym Setup", "Protein Diet Plan"],
    "food": ["Pasta Recipe in 10 Min", "Street Food Tour Mumbai", "Biryani Secrets", "Healthy Breakfast Ideas", "5-Ingredient Desserts"],
    "comedy": ["Office Life Be Like", "Mom vs Kids", "When WiFi Stops Working", "Exam Season Mood", "Autowaala Stories"],
    "love": ["Propose Karo Aise", "Long Distance Tips", "First Date Red Flags", "Couple Goals 2026", "True Love Signs"],
    "news": ["Breaking: Market Update", "Political Drama Today", "Weather Alert India", "World News Roundup", "Economy Report"],
    "motivation": ["Wake Up at 5AM", "Your Dream Won't Wait", "No Excuses Mode", "Level Up Daily", "Discipline Over Motivation"],
    "quotes": ["Life Is Too Short", "Find Your Purpose", "Silence Is Power", "Trust The Process", "Be The Change"],
    "cricket": ["IPL 2026 Highlights", "Virat's Best Shots", "India vs Pakistan Recap", "Bowling Masterclass", "Fantasy Team Tips"],
    "tech": ["AI Will Change Everything", "Python in 10 Minutes", "Best Gadgets 2026", "Coding Tips for Freshers", "ChatGPT vs Claude"],
    "music": ["Lofi Beats to Study", "Top Bollywood 2026", "Guitar for Beginners", "Music Therapy Works", "New Album Review"],
}

SATURATION_THRESHOLD = 0.40
MAX_WINDOW = 30
DATASET_FILE = "feedbalance_dataset.csv"
HISTORY_FILE = "watch_history.json"


def parse_dotenv(path=".env"):
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_iso8601_duration(duration):
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def guess_category(title):
    text = title.lower()
    keywords = {
        "fitness": ["workout", "gym", "protein", "weight", "fitness"],
        "food": ["food", "recipe", "biryani", "breakfast", "dessert", "pasta"],
        "comedy": ["funny", "comedy", "meme", "joke", "standup"],
        "love": ["love", "date", "couple", "relationship", "romantic"],
        "news": ["news", "breaking", "market", "economy", "politics"],
        "motivation": ["motivation", "discipline", "dream", "success", "level up"],
        "quotes": ["quote", "purpose", "silence", "process"],
        "cricket": ["cricket", "ipl", "virat", "india vs", "bowling"],
        "tech": ["ai", "python", "coding", "gadget", "tech"],
        "music": ["music", "song", "album", "guitar", "lofi"],
    }
    for category, words in keywords.items():
        if any(word in text for word in words):
            return category
    return random.choice(CATEGORIES)


def generate_fake_dataset(n=120, bias_category=None, bias_strength=0.45):
    rows = []
    base_time = datetime.now() - timedelta(days=14)
    for idx in range(n):
        category = bias_category if bias_category and random.random() < bias_strength else random.choice(CATEGORIES)
        watch_duration = random.randint(15, 240)
        views = random.randint(1_000, 900_000)
        likes = random.randint(20, max(25, int(views * 0.08)))
        comments = random.randint(0, max(5, int(views * 0.01)))
        rows.append({
            "content_id": f"demo_{idx + 1:04d}",
            "title": random.choice(SAMPLE_TITLES[category]),
            "category": category,
            "watch_duration_sec": watch_duration,
            "liked": random.random() > 0.52,
            "shared": random.random() > 0.82,
            "views": views,
            "likes": likes,
            "comments": comments,
            "engagement_rate": round((likes + comments) / views, 4),
            "timestamp": (base_time + timedelta(minutes=random.randint(0, 14 * 24 * 60))).isoformat(timespec="minutes"),
            "source": "synthetic",
        })
    return pd.DataFrame(rows)


def load_phase1_history(path=HISTORY_FILE):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df.rename(columns={"id": "content_id", "watch_time": "watch_duration_sec"})
    df["content_id"] = df["content_id"].apply(lambda value: f"phase1_{value}")
    df["views"] = np.random.randint(1_000, 250_000, size=len(df))
    df["likes"] = np.random.randint(20, 12_000, size=len(df))
    df["comments"] = np.random.randint(0, 1_200, size=len(df))
    df["engagement_rate"] = ((df["likes"] + df["comments"]) / df["views"]).round(4)
    df["source"] = "phase1_json"
    return df


def fetch_youtube_data(api_key, max_results=25, region_code="IN"):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "chart": "mostPopular",
        "regionCode": region_code,
        "maxResults": max_results,
        "key": api_key,
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    rows = []
    for item in response.json().get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        title = snippet.get("title", "Untitled")
        rows.append({
            "content_id": item.get("id"),
            "title": title,
            "category": guess_category(title),
            "watch_duration_sec": parse_iso8601_duration(details.get("duration")),
            "liked": likes > 0,
            "shared": False,
            "views": views,
            "likes": likes,
            "comments": comments,
            "engagement_rate": round((likes + comments) / views, 4) if views else 0.0,
            "timestamp": snippet.get("publishedAt"),
            "source": "youtube_api",
        })
    return pd.DataFrame(rows)


def ensure_columns(df):
    if df.empty:
        return generate_fake_dataset()
    result = df.copy()
    for column, default in {
        "content_id": "unknown",
        "title": "Untitled",
        "category": "unknown",
        "watch_duration_sec": 0,
        "liked": False,
        "shared": False,
        "views": 0,
        "likes": 0,
        "comments": 0,
        "engagement_rate": 0.0,
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "source": "unknown",
    }.items():
        if column not in result.columns:
            result[column] = default
    result["watch_duration_sec"] = pd.to_numeric(result["watch_duration_sec"], errors="coerce").fillna(0).astype(int)
    result["views"] = pd.to_numeric(result["views"], errors="coerce").fillna(0).astype(int)
    result["likes"] = pd.to_numeric(result["likes"], errors="coerce").fillna(0).astype(int)
    result["comments"] = pd.to_numeric(result["comments"], errors="coerce").fillna(0).astype(int)
    result["engagement_rate"] = pd.to_numeric(result["engagement_rate"], errors="coerce").fillna(0.0)
    return result


def numpy_category_analysis(df, threshold=SATURATION_THRESHOLD):
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    values = counts.to_numpy(dtype=float)
    total = values.sum()
    shares = values / total if total else np.zeros_like(values)
    active = shares[shares > 0]
    entropy = -np.sum(active * np.log2(active)) if len(active) else 0.0
    diversity_score = entropy / math.log2(len(active)) if len(active) > 1 else 0.0
    saturated_idx = np.where(shares >= threshold)[0]
    top_idx = int(np.argmax(shares)) if len(shares) else 0
    return {
        "counts": counts,
        "shares": pd.Series(shares, index=CATEGORIES),
        "top_category": CATEGORIES[top_idx],
        "top_share": float(shares[top_idx]) if len(shares) else 0.0,
        "diversity_score": float(round(diversity_score, 3)),
        "saturated_cats": [CATEGORIES[i] for i in saturated_idx],
        "mean_count": float(np.mean(values)),
        "std_count": float(np.std(values)),
    }


def pandas_analysis(df):
    category_summary = (
        df.groupby("category")
        .agg(
            videos=("content_id", "count"),
            avg_watch_sec=("watch_duration_sec", "mean"),
            total_watch_sec=("watch_duration_sec", "sum"),
            like_rate=("liked", "mean"),
            share_rate=("shared", "mean"),
            avg_engagement=("engagement_rate", "mean"),
            total_views=("views", "sum"),
        )
        .sort_values("videos", ascending=False)
        .round(3)
    )
    describe = df[["watch_duration_sec", "views", "likes", "comments", "engagement_rate"]].describe().round(2)
    long_watch = df[df["watch_duration_sec"] >= df["watch_duration_sec"].quantile(0.75)]
    return {
        "category_summary": category_summary,
        "describe": describe,
        "long_watch": long_watch,
        "source_counts": df["source"].value_counts(),
    }


def save_dataset(df, path=DATASET_FILE):
    df.to_csv(path, index=False)


@st.cache_data(show_spinner=False)
def cached_fake_dataset(n, bias_category, bias_strength):
    return generate_fake_dataset(n, bias_category, bias_strength)


with st.sidebar:
    st.title("FeedBalance")
    st.caption("Phase 2 - Data Science")
    st.divider()

    source = st.radio("Dataset source", ["Synthetic", "Phase 1 JSON", "Saved CSV", "YouTube API"], index=0)
    n_items = st.slider("Synthetic rows", 30, 300, 120, 10)
    bias_options = ["None"] + CATEGORIES
    bias_category = st.selectbox("Synthetic bias", bias_options)
    bias_category = None if bias_category == "None" else bias_category
    bias_strength = st.slider("Bias strength", 0.10, 0.80, 0.45, 0.05)
    threshold = st.slider("Saturation threshold", 0.20, 0.70, SATURATION_THRESHOLD, 0.05)

    st.divider()
    dotenv_key = parse_dotenv().get("YOUTUBE_API_KEY")
    api_key = os.environ.get("YOUTUBE_API_KEY") or dotenv_key or ""
    max_results = st.slider("YouTube results", 5, 50, 25, 5)
    region_code = st.text_input("YouTube region", "IN", max_chars=2).upper()


if source == "Synthetic":
    df = cached_fake_dataset(n_items, bias_category, bias_strength)
elif source == "Phase 1 JSON":
    df = load_phase1_history()
    if df.empty:
        st.warning("watch_history.json was not found or had no rows, so a synthetic dataset is loaded.")
        df = cached_fake_dataset(n_items, bias_category, bias_strength)
elif source == "Saved CSV":
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
    else:
        st.warning("feedbalance_dataset.csv does not exist yet, so a synthetic dataset is loaded.")
        df = cached_fake_dataset(n_items, bias_category, bias_strength)
else:
    if api_key:
        try:
            df = fetch_youtube_data(api_key, max_results=max_results, region_code=region_code)
        except requests.RequestException as exc:
            st.error(f"YouTube API fetch failed: {exc}")
            df = cached_fake_dataset(n_items, bias_category, bias_strength)
    else:
        st.warning("Set YOUTUBE_API_KEY in .env or environment variables. Synthetic data loaded for now.")
        df = cached_fake_dataset(n_items, bias_category, bias_strength)

df = ensure_columns(df)
np_report = numpy_category_analysis(df, threshold)
pd_report = pandas_analysis(df)

st.title("FeedBalance - Phase 2")
st.caption("NumPy, Pandas, Matplotlib/Seaborn, CSV output, and optional YouTube Data API ingestion.")

metric_cols = st.columns(5)
metric_cols[0].metric("Rows", len(df))
metric_cols[1].metric("Top Category", np_report["top_category"], f"{np_report['top_share']:.0%}")
metric_cols[2].metric("Diversity", f"{np_report['diversity_score']:.2f} / 1.0")
metric_cols[3].metric("Saturated", ", ".join(np_report["saturated_cats"]) if np_report["saturated_cats"] else "None")
metric_cols[4].metric("Avg Watch", f"{df['watch_duration_sec'].mean():.0f}s")

tab_dashboard, tab_explorer, tab_charts, tab_learn, tab_code = st.tabs([
    "Dashboard",
    "Data Explorer",
    "Charts",
    "Learn",
    "Code",
])

with tab_dashboard:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Category summary")
        st.dataframe(pd_report["category_summary"], use_container_width=True)
    with right:
        st.subheader("Saturation status")
        if np_report["saturated_cats"]:
            st.error(f"{', '.join(np_report['saturated_cats'])} crossed the {threshold:.0%} threshold.")
        else:
            st.success(f"No category crossed the {threshold:.0%} threshold.")
        st.write("NumPy signals")
        st.json({
            "mean_count": round(np_report["mean_count"], 2),
            "std_count": round(np_report["std_count"], 2),
            "diversity_score": np_report["diversity_score"],
        })

    if st.button("Save current dataset for Phase 3", use_container_width=True):
        save_dataset(df)
        st.success(f"Saved {len(df)} rows to {DATASET_FILE}")

with tab_explorer:
    selected_categories = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
    min_watch = st.slider("Minimum watch seconds", 0, int(max(df["watch_duration_sec"].max(), 1)), 0)
    filtered = df[df["category"].isin(selected_categories) & (df["watch_duration_sec"] >= min_watch)]
    st.dataframe(filtered, use_container_width=True, height=460)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "feedbalance_filtered.csv",
        "text/csv",
        use_container_width=True,
    )

with tab_charts:
    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = np_report["counts"].sort_values()
        ax.barh(counts.index, counts.values, color=[CAT_COLOR.get(cat, "#64748b") for cat in counts.index])
        ax.set_title("Category counts")
        ax.set_xlabel("Videos")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="category", y="watch_duration_sec", ax=ax, color="#60a5fa")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Watch duration spread")
        st.pyplot(fig)
        plt.close(fig)

    with chart_col_2:
        time_df = df.copy()
        time_df["timestamp"] = pd.to_datetime(time_df["timestamp"], errors="coerce")
        daily = time_df.dropna(subset=["timestamp"]).set_index("timestamp").resample("D").size()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(daily.index, daily.values, marker="o", color="#10b981")
        ax.set_title("Daily watch volume")
        ax.set_ylabel("Videos")
        st.pyplot(fig)
        plt.close(fig)

        pivot = pd.crosstab(df["category"], df["source"])
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Category by data source")
        st.pyplot(fig)
        plt.close(fig)

with tab_learn:
    st.subheader("Phase 2 concepts")
    st.markdown("""
    | Concept | Where it appears |
    |---|---|
    | NumPy arrays | Category counts, shares, entropy, threshold detection |
    | Pandas DataFrames | Filtering, grouping, aggregation, describe, CSV export |
    | Matplotlib/Seaborn | Bar chart, line chart, boxplot, heatmap |
    | API ingestion | YouTube Data API v3 with duration parsing and engagement metrics |
    | Phase 3 bridge | `feedbalance_dataset.csv` contains `title` + `category` for ML classification |
    """)

with tab_code:
    st.subheader("Run")
    st.code("streamlit run feedbalance_phase2.py", language="bash")
    st.subheader("Phase 3-ready columns")
    st.code(", ".join(df.columns), language="text")
    st.subheader("Pandas describe")
    st.dataframe(pd_report["describe"], use_container_width=True)

st.divider()
st.caption("FeedBalance Phase 2 - NumPy - Pandas - Charts - YouTube API - CSV dataset for Phase 3")
