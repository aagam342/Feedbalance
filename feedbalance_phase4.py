"""FeedBalance Phase 4: SQLite-backed diversity recommendation engine."""

import math
import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st


DB_FILE = "feedbalance.db"
DATASET_FILE = "feedbalance_dataset.csv"
SATURATION_THRESHOLD = 0.40
MAX_WINDOW = 30
DEFAULT_USER_ID = "demo_user"

CATEGORIES = [
    "fitness", "food", "comedy", "love", "news",
    "motivation", "quotes", "cricket", "tech", "music",
]


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def bool_to_int(value):
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))


def shannon_entropy(shares):
    active = [share for share in shares.values() if share > 0]
    if len(active) <= 1:
        return 0.0
    entropy = -sum(share * math.log2(share) for share in active)
    return round(entropy / math.log2(len(active)), 3)


class FeedBalanceEngine:
    def __init__(self, db_path=DB_FILE, threshold=SATURATION_THRESHOLD, max_window=MAX_WINDOW):
        self.db_path = db_path
        self.threshold = threshold
        self.max_window = max_window
        self.init_db()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    title TEXT,
                    category TEXT NOT NULL,
                    emotional_tone TEXT,
                    watch_duration_sec INTEGER DEFAULT 0,
                    liked INTEGER DEFAULT 0,
                    shared INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    title TEXT,
                    category TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS saturation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    top_category TEXT,
                    saturation_score REAL NOT NULL,
                    diversity_score REAL NOT NULL,
                    action_taken TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

    def ensure_user(self, user_id=DEFAULT_USER_ID):
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
                (user_id, utc_now()),
            )

    def ingest(self, df, user_id=DEFAULT_USER_ID, replace_user_history=True):
        self.ensure_user(user_id)
        clean = df.copy()
        clean = clean.dropna(subset=["content_id", "category"])
        if "title" not in clean.columns:
            clean["title"] = clean["content_id"].astype(str)
        if "watch_duration_sec" not in clean.columns:
            clean["watch_duration_sec"] = 0
        if "liked" not in clean.columns:
            clean["liked"] = False
        if "shared" not in clean.columns:
            clean["shared"] = False
        if "timestamp" not in clean.columns:
            clean["timestamp"] = utc_now()
        if "source" not in clean.columns:
            clean["source"] = "unknown"

        with self.connect() as conn:
            if replace_user_history:
                conn.execute("DELETE FROM watch_history WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM recommendations WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM saturation_events WHERE user_id = ?", (user_id,))

            rows = []
            for _, row in clean.iterrows():
                rows.append((
                    user_id,
                    str(row["content_id"]),
                    str(row.get("title", "")),
                    str(row["category"]).lower(),
                    str(row.get("emotional_tone", "")) or None,
                    int(row.get("watch_duration_sec", 0)),
                    bool_to_int(row.get("liked", False)),
                    bool_to_int(row.get("shared", False)),
                    str(row.get("timestamp", utc_now())),
                    str(row.get("source", "unknown")),
                ))
            conn.executemany("""
                INSERT INTO watch_history (
                    user_id, content_id, title, category, emotional_tone,
                    watch_duration_sec, liked, shared, timestamp, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

        status = self.get_status(user_id)
        if status["saturated"]:
            self.record_saturation_event(user_id, status, "diversity_injection")
        return len(rows)

    def get_recent_history(self, user_id=DEFAULT_USER_ID):
        query = """
            SELECT *
            FROM watch_history
            WHERE user_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=(user_id, self.max_window))

    def get_status(self, user_id=DEFAULT_USER_ID):
        history = self.get_recent_history(user_id)
        if history.empty:
            return {
                "total": 0,
                "counts": {},
                "shares": {},
                "top_category": None,
                "top_share": 0.0,
                "diversity_score": 0.0,
                "saturated": False,
                "saturated_categories": [],
            }

        counts = history["category"].value_counts().to_dict()
        total = int(sum(counts.values()))
        shares = {cat: count / total for cat, count in counts.items()}
        top_category = max(shares, key=shares.get)
        saturated_categories = [cat for cat, share in shares.items() if share >= self.threshold]
        return {
            "total": total,
            "counts": counts,
            "shares": shares,
            "top_category": top_category,
            "top_share": shares[top_category],
            "diversity_score": shannon_entropy(shares),
            "saturated": len(saturated_categories) > 0,
            "saturated_categories": saturated_categories,
        }

    def record_saturation_event(self, user_id, status, action_taken):
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO saturation_events (
                    user_id, timestamp, top_category, saturation_score,
                    diversity_score, action_taken
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                utc_now(),
                status["top_category"],
                float(status["top_share"]),
                float(status["diversity_score"]),
                action_taken,
            ))

    def get_candidate_pool(self, path=DATASET_FILE):
        if not os.path.exists(path):
            return pd.DataFrame(columns=["content_id", "title", "category", "watch_duration_sec"])
        df = pd.read_csv(path)
        required = {"content_id", "title", "category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Candidate dataset missing columns: {', '.join(sorted(missing))}")
        return df

    def get_recs(self, user_id=DEFAULT_USER_ID, candidate_df=None, limit=10):
        status = self.get_status(user_id)
        if candidate_df is None:
            candidate_df = self.get_candidate_pool()
        if candidate_df.empty:
            return pd.DataFrame(columns=["content_id", "title", "category", "score", "reason"])

        with self.connect() as conn:
            seen = pd.read_sql_query(
                "SELECT DISTINCT content_id FROM watch_history WHERE user_id = ?",
                conn,
                params=(user_id,),
            )
        seen_ids = set(seen["content_id"].astype(str))
        candidates = candidate_df[~candidate_df["content_id"].astype(str).isin(seen_ids)].copy()
        if candidates.empty:
            candidates = candidate_df.copy()

        saturated = set(status["saturated_categories"])
        shares = status["shares"]
        rows = []
        for _, row in candidates.iterrows():
            category = str(row["category"]).lower()
            share = shares.get(category, 0.0)
            exploration_boost = max(0.0, self.threshold - share)
            score = 1.0 + exploration_boost
            reason = "Balanced exploration"

            if category in saturated:
                score -= 0.75
                reason = f"Limited because {category} is saturated"
            elif status["saturated"]:
                score += 0.65
                reason = f"Boosted to offset {status['top_category']} saturation"
            elif share == 0:
                score += 0.35
                reason = "Boosted as an unseen category"

            engagement = float(row.get("engagement_rate", 0.0) or 0.0)
            score += min(0.25, engagement)
            rows.append({
                "content_id": str(row["content_id"]),
                "title": str(row.get("title", "")),
                "category": category,
                "score": round(score, 4),
                "reason": reason,
            })

        recs = pd.DataFrame(rows).sort_values("score", ascending=False).head(limit)
        self.save_recommendations(user_id, recs)
        return recs

    def save_recommendations(self, user_id, recs):
        if recs.empty:
            return
        generated_at = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM recommendations WHERE user_id = ?", (user_id,))
            conn.executemany("""
                INSERT INTO recommendations (
                    user_id, content_id, title, category, score, reason, generated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    user_id,
                    row["content_id"],
                    row["title"],
                    row["category"],
                    float(row["score"]),
                    row["reason"],
                    generated_at,
                )
                for _, row in recs.iterrows()
            ])

    def analytics(self, user_id=DEFAULT_USER_ID):
        with self.connect() as conn:
            tables = {
                "users": pd.read_sql_query("SELECT * FROM users", conn),
                "watch_history": pd.read_sql_query(
                    "SELECT * FROM watch_history WHERE user_id = ? ORDER BY timestamp DESC, id DESC",
                    conn,
                    params=(user_id,),
                ),
                "recommendations": pd.read_sql_query(
                    "SELECT * FROM recommendations WHERE user_id = ? ORDER BY score DESC",
                    conn,
                    params=(user_id,),
                ),
                "saturation_events": pd.read_sql_query(
                    "SELECT * FROM saturation_events WHERE user_id = ? ORDER BY timestamp DESC",
                    conn,
                    params=(user_id,),
                ),
            }
        return tables


def load_dataset(path=DATASET_FILE):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    st.set_page_config(
        page_title="FeedBalance Phase 4",
        page_icon="DB",
        layout="wide",
    )

    st.title("FeedBalance - Phase 4")
    st.caption("SQLite backend plus diversity-aware recommendation engine.")

    with st.sidebar:
        st.title("FeedBalance")
        st.caption("Phase 4 - Recommendation Engine")
        st.divider()
        user_id = st.text_input("User ID", DEFAULT_USER_ID)
        threshold = st.slider("Saturation threshold", 0.20, 0.70, SATURATION_THRESHOLD, 0.05)
        max_window = st.slider("Rolling window", 10, 100, MAX_WINDOW, 5)
        rec_limit = st.slider("Recommendations", 5, 20, 10, 1)
        replace_history = st.checkbox("Replace user history on ingest", value=True)
        st.divider()
        st.code("streamlit run feedbalance_phase4.py", language="bash")

    engine = FeedBalanceEngine(threshold=threshold, max_window=max_window)
    dataset = load_dataset()

    if dataset.empty:
        st.error("No feedbalance_dataset.csv found. Complete Phase 2 first.")
        st.stop()

    if st.button("Ingest Phase 2 dataset into SQLite", use_container_width=True):
        rows = engine.ingest(dataset, user_id=user_id, replace_user_history=replace_history)
        st.success(f"Ingested {rows} rows into {DB_FILE}")

    existing = engine.analytics(user_id)["watch_history"]
    if existing.empty:
        rows = engine.ingest(dataset, user_id=user_id, replace_user_history=True)
        st.info(f"Initialized demo database with {rows} rows.")

    status = engine.get_status(user_id)
    recs = engine.get_recs(user_id=user_id, candidate_df=dataset, limit=rec_limit)
    analytics = engine.analytics(user_id)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Recent Events", status["total"])
    metric_cols[1].metric("Top Category", status["top_category"] or "None", f"{status['top_share']:.0%}")
    metric_cols[2].metric("Diversity", f"{status['diversity_score']:.2f} / 1.0")
    metric_cols[3].metric("Status", "Saturated" if status["saturated"] else "Healthy")
    metric_cols[4].metric("DB Rows", len(analytics["watch_history"]))

    tab_status, tab_recs, tab_db, tab_learn = st.tabs([
        "Status",
        "Recommendations",
        "Database",
        "Learn",
    ])

    with tab_status:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Current feed balance")
            if status["saturated"]:
                st.error(f"{', '.join(status['saturated_categories'])} crossed {threshold:.0%}.")
            else:
                st.success(f"No category crossed {threshold:.0%}.")
            st.json({
                "counts": status["counts"],
                "shares": {cat: round(share, 3) for cat, share in status["shares"].items()},
            })
        with right:
            st.subheader("Recent watch history")
            st.dataframe(analytics["watch_history"].head(max_window), use_container_width=True, height=420)

    with tab_recs:
        st.subheader("Diversity-aware recommendations")
        st.dataframe(recs, use_container_width=True, height=420)
        st.download_button(
            "Download recommendations CSV",
            recs.to_csv(index=False).encode("utf-8"),
            "feedbalance_recommendations.csv",
            "text/csv",
            use_container_width=True,
        )

    with tab_db:
        st.subheader("SQLite tables")
        db_tabs = st.tabs(["users", "watch_history", "recommendations", "saturation_events"])
        for tab, name in zip(db_tabs, ["users", "watch_history", "recommendations", "saturation_events"]):
            with tab:
                st.dataframe(analytics[name], use_container_width=True, height=360)

    with tab_learn:
        st.subheader("Phase 4 concepts")
        st.markdown("""
        | Component | Implementation |
        |---|---|
        | SQLite backend | `feedbalance.db` with users, watch_history, recommendations, saturation_events |
        | Engine class | `FeedBalanceEngine.ingest/get_status/get_recs/analytics` |
        | Rolling window | Last `MAX_WINDOW` watch events drive status |
        | Saturation event | Stored whenever a category crosses threshold |
        | Diversity injection | Saturated categories are downweighted; missing or underrepresented categories are boosted |
        | Phase 5 bridge | `emotional_tone` column exists for future NLP classifier output |
        """)

    st.divider()
    st.caption("FeedBalance Phase 4 - SQLite - Recommendation Engine - Diversity-aware ranking")


if __name__ == "__main__":
    main()
