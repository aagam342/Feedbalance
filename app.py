"""FeedBalance production Streamlit app for Phase 6 deployment."""

import os
import sqlite3

import joblib
import pandas as pd
import streamlit as st

from feedbalance_phase4 import (
    DB_FILE,
    DEFAULT_USER_ID,
    FeedBalanceEngine,
    load_dataset,
)
from feedbalance_phase5 import (
    MODEL_FILE as TONE_MODEL_FILE,
    clean_text,
    load_dataset as load_tone_dataset,
    save_model as save_tone_model,
    train_tone_classifier,
    update_database_tones,
)


CATEGORY_MODEL_FILE = "category_classifier.pkl"


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --fb-ink: #0f172a;
            --fb-muted: #64748b;
            --fb-line: #dbe3ef;
            --fb-surface: #ffffff;
            --fb-band: #eef6ff;
            --fb-blue: #2563eb;
            --fb-cyan: #0891b2;
            --fb-green: #059669;
            --fb-red: #dc2626;
            --fb-amber: #b45309;
        }

        .stApp {
            background: #f8fafc;
            color: var(--fb-ink);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--fb-line);
        }

        section[data-testid="stSidebar"] * {
            color: var(--fb-ink);
        }

        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #ffffff;
            color: #0f172a;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        .fb-hero {
            background:
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.13), transparent 34%),
                linear-gradient(135deg, #ffffff 0%, #edf7ff 100%);
            border: 1px solid #cfe0f3;
            border-radius: 8px;
            padding: 24px 26px;
            margin-bottom: 18px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        }

        .fb-hero h1 {
            color: var(--fb-ink);
            margin: 0 0 8px 0;
            font-size: 2.35rem;
            letter-spacing: 0;
        }

        .fb-hero p {
            color: #475569;
            max-width: 780px;
            margin: 0;
            font-size: 1rem;
            line-height: 1.55;
        }

        .fb-status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .fb-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            background: #eff6ff;
        }

        .fb-metric-card {
            min-height: 126px;
            height: 126px;
            background: var(--fb-surface);
            border: 1px solid var(--fb-line);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .fb-metric-label {
            color: var(--fb-muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .fb-metric-value {
            color: var(--fb-ink);
            font-size: 1.95rem;
            line-height: 1.1;
            font-weight: 750;
            overflow-wrap: anywhere;
        }

        .fb-metric-note {
            color: #047857;
            font-size: 0.82rem;
            font-weight: 700;
            min-height: 18px;
        }

        .fb-panel {
            background: var(--fb-surface);
            border: 1px solid var(--fb-line);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }

        .fb-panel h3 {
            margin: 0 0 10px 0;
            font-size: 1.05rem;
            color: var(--fb-ink);
        }

        .fb-panel p {
            color: var(--fb-muted);
            line-height: 1.5;
        }

        .fb-callout {
            border-left: 4px solid var(--fb-blue);
            background: var(--fb-band);
            padding: 13px 14px;
            border-radius: 8px;
            color: #164e63;
            margin: 8px 0 14px 0;
        }

        .fb-callout-danger {
            border-left-color: var(--fb-red);
            background: #fff1f2;
            color: #991b1b;
        }

        .fb-callout-good {
            border-left-color: var(--fb-green);
            background: #ecfdf5;
            color: #065f46;
        }

        .fb-result {
            background: #ffffff;
            border: 1px solid var(--fb-line);
            border-top: 4px solid var(--fb-blue);
            border-radius: 8px;
            padding: 16px;
            min-height: 112px;
        }

        .fb-result .label {
            color: var(--fb-muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .fb-result .value {
            color: var(--fb-ink);
            font-size: 1.9rem;
            line-height: 1.2;
            margin-top: 8px;
            font-weight: 700;
        }

        .fb-file-list {
            display: grid;
            gap: 8px;
            margin-top: 8px;
        }

        .fb-file-item {
            border: 1px solid var(--fb-line);
            background: #f8fafc;
            border-radius: 8px;
            padding: 10px 11px;
            font-size: 0.84rem;
        }

        .fb-file-item b {
            display: block;
            margin-bottom: 4px;
            color: var(--fb-ink);
        }

        .fb-file-item span {
            color: var(--fb-muted);
            overflow-wrap: anywhere;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def load_pickle(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def model_file_label(path):
    return path if os.path.exists(path) else f"{path} missing"


def metric_card(label, value, note=""):
    return f"""
    <div class="fb-metric-card">
        <div class="fb-metric-label">{label}</div>
        <div class="fb-metric-value">{value}</div>
        <div class="fb-metric-note">{note}</div>
    </div>
    """


def ensure_production_state(user_id=DEFAULT_USER_ID):
    engine = FeedBalanceEngine()
    dataset = load_dataset()
    analytics = engine.analytics(user_id)
    if analytics["watch_history"].empty and not dataset.empty:
        engine.ingest(dataset, user_id=user_id, replace_user_history=True)

    if not os.path.exists(TONE_MODEL_FILE) and not dataset.empty:
        tone_df = load_tone_dataset()
        result = train_tone_classifier(tone_df)
        save_tone_model(result["pipeline"])

    tone_model = load_pickle(TONE_MODEL_FILE)
    if tone_model is not None and not dataset.empty:
        tone_df = load_tone_dataset()
        update_database_tones(tone_df, tone_model, user_id=user_id)
    return engine, dataset, tone_model


def read_table(name, user_id=DEFAULT_USER_ID):
    with sqlite3.connect(DB_FILE) as conn:
        if name == "users":
            return pd.read_sql_query("SELECT * FROM users", conn)
        return pd.read_sql_query(
            f"SELECT * FROM {name} WHERE user_id = ?",
            conn,
            params=(user_id,),
        )


def tone_distribution(user_id=DEFAULT_USER_ID):
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            """
            SELECT emotional_tone, COUNT(*) AS rows
            FROM watch_history
            WHERE user_id = ? AND emotional_tone IS NOT NULL
            GROUP BY emotional_tone
            ORDER BY rows DESC
            """,
            conn,
            params=(user_id,),
        )


def main():
    st.set_page_config(
        page_title="FeedBalance",
        page_icon="FB",
        layout="wide",
    )
    inject_styles()

    with st.sidebar:
        st.title("FeedBalance")
        st.caption("Content wellbeing dashboard")
        profile = st.selectbox("Profile", ["Demo feed profile"], index=0)
        user_id = DEFAULT_USER_ID
        st.caption("Uses the bundled demo watch history and model artifacts.")
        rec_limit = st.slider("Recommendations", 5, 20, 10, 1)
        st.divider()
        st.caption("Model files")
        st.markdown(
            f"""
            <div class="fb-file-list">
                <div class="fb-file-item"><b>Category classifier</b><span>{model_file_label(CATEGORY_MODEL_FILE)}</span></div>
                <div class="fb-file-item"><b>Emotional tone classifier</b><span>{model_file_label(TONE_MODEL_FILE)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    engine, dataset, tone_model = ensure_production_state(user_id)
    status = engine.get_status(user_id)
    recs = engine.get_recs(user_id=user_id, candidate_df=dataset, limit=rec_limit)
    analytics = engine.analytics(user_id)
    tone_dist = tone_distribution(user_id)
    category_model = load_pickle(CATEGORY_MODEL_FILE)

    state_label = "Saturated" if status["saturated"] else "Balanced"
    model_label = "Models ready" if category_model is not None and tone_model is not None else "Model check needed"
    st.markdown(
        f"""
        <div class="fb-hero">
            <h1>FeedBalance</h1>
            <p>Mental health-aware content diversity system that detects feed saturation, recommends balancing content, and predicts emotional tone before the feed becomes repetitive.</p>
            <div class="fb-status-row">
                <span class="fb-pill">Status: {state_label}</span>
                <span class="fb-pill">Category: {model_file_label(CATEGORY_MODEL_FILE)}</span>
                <span class="fb-pill">Tone: {model_file_label(TONE_MODEL_FILE)}</span>
                <span class="fb-pill">{model_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].markdown(metric_card("Watch Events", len(analytics["watch_history"]), "SQLite history"), unsafe_allow_html=True)
    metric_cols[1].markdown(metric_card("Top Category", status["top_category"] or "None", f'{status["top_share"]:.0%} recent share'), unsafe_allow_html=True)
    metric_cols[2].markdown(metric_card("Diversity", f'{status["diversity_score"]:.2f} / 1.0', "higher is better"), unsafe_allow_html=True)
    metric_cols[3].markdown(metric_card("Recommendations", len(recs), "balanced ranking"), unsafe_allow_html=True)
    metric_cols[4].markdown(metric_card("Tone Labels", len(tone_dist), "NLP layer"), unsafe_allow_html=True)

    tab_overview, tab_recs, tab_predict, tab_data = st.tabs([
        "Overview",
        "Recommendations",
        "Predict",
        "Data",
    ])

    with tab_overview:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Feed status")
            if status["saturated"]:
                st.markdown(
                    f'<div class="fb-callout fb-callout-danger">{", ".join(status["saturated_categories"])} crossed the saturation threshold. FeedBalance will down-rank this category and boost variety.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="fb-callout fb-callout-good">Feed is balanced. No category is dominating the recent watch window.</div>',
                    unsafe_allow_html=True,
                )
            share_df = pd.DataFrame({
                "category": list(status["shares"].keys()),
                "share": [round(value, 3) for value in status["shares"].values()],
            }).sort_values("share", ascending=False)
            st.dataframe(
                share_df,
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.subheader("Emotional tone mix")
            st.bar_chart(tone_dist.set_index("emotional_tone") if not tone_dist.empty else pd.DataFrame())

    with tab_recs:
        st.markdown(
            '<div class="fb-callout">Recommendations are ranked by category balance first, then lightly adjusted by engagement. Saturated categories are intentionally reduced.</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Diversity-aware recommendations")
        st.dataframe(recs, use_container_width=True, height=420)
        st.download_button(
            "Download recommendations",
            recs.to_csv(index=False).encode("utf-8"),
            "feedbalance_recommendations.csv",
            "text/csv",
            use_container_width=True,
        )

    with tab_predict:
        st.subheader("Live content analysis")
        title = st.text_input("Content title", "Python coding tips for beginners")
        if title:
            cols = st.columns(2)
            if category_model is not None:
                category = category_model.predict([clean_text(title)])[0]
                cols[0].markdown(
                    f'<div class="fb-result"><div class="label">Predicted Category</div><div class="value">{category}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                cols[0].warning("Category model file is missing.")

            if tone_model is not None:
                tone = tone_model.predict([clean_text(title)])[0]
                cols[1].markdown(
                    f'<div class="fb-result"><div class="label">Predicted Emotional Tone</div><div class="value">{tone}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                cols[1].warning("Tone model file is missing.")

    with tab_data:
        st.subheader("SQLite-backed analytics")
        db_tabs = st.tabs(["watch_history", "recommendations", "saturation_events", "tone_events"])
        for db_tab, table in zip(db_tabs, ["watch_history", "recommendations", "saturation_events", "tone_events"]):
            with db_tab:
                try:
                    st.dataframe(read_table(table, user_id), use_container_width=True, height=380)
                except Exception as exc:
                    st.warning(f"{table} is not available yet: {exc}")

    st.divider()
    st.caption("FeedBalance Phase 6 - deployment-ready Streamlit app")


if __name__ == "__main__":
    main()
