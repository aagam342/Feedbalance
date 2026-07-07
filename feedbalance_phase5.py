"""FeedBalance Phase 5: emotional-tone NLP layer."""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_FILE = "feedbalance_dataset.csv"
DB_FILE = "feedbalance.db"
MODEL_FILE = "emotional_tone_classifier.pkl"
TRACKING_FILE = "phase5_tone_metrics.json"
DEFAULT_USER_ID = "demo_user"

TONES = [
    "Aspirational",
    "Humorous",
    "Melancholic",
    "Outrage",
    "Educational",
    "Romantic",
    "Relaxing",
]


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def clean_text(value):
    return str(value).strip().lower()


def dependency_status():
    status = {}
    for name in ["torch", "transformers", "mlflow"]:
        try:
            __import__(name)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


def infer_tone_from_row(row):
    title = clean_text(row.get("title", ""))
    category = clean_text(row.get("category", ""))

    humorous = ["funny", "comedy", "be like", "wifi", "mom vs", "autowaala", "exam season"]
    romantic = ["love", "date", "couple", "propose", "relationship", "red flags"]
    aspirational = ["dream", "discipline", "motivation", "level up", "purpose", "process", "workout", "fitness", "virat"]
    outrage = ["breaking", "drama", "alert", "political", "market update"]
    educational = ["python", "coding", "tips", "ai", "tech", "recipe", "secrets", "masterclass", "report", "gadget"]
    relaxing = ["lofi", "music", "guitar", "therapy", "breakfast", "dessert", "pasta"]
    melancholic = ["silence", "long distance", "life is too short"]

    if any(word in title for word in humorous) or category == "comedy":
        return "Humorous"
    if any(word in title for word in romantic) or category == "love":
        return "Romantic"
    if any(word in title for word in outrage) or category == "news":
        return "Outrage"
    if any(word in title for word in relaxing) or category in {"music", "food"}:
        return "Relaxing"
    if any(word in title for word in melancholic):
        return "Melancholic"
    if any(word in title for word in educational) or category == "tech":
        return "Educational"
    if any(word in title for word in aspirational) or category in {"motivation", "quotes", "fitness", "cricket"}:
        return "Aspirational"
    return "Educational"


def load_dataset(path=DATASET_FILE):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["content_id", "title", "category"])

    df = pd.read_csv(path)
    required = {"content_id", "title", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["content_id", "title", "category"]).copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    if "emotional_tone" not in df.columns:
        df["emotional_tone"] = df.apply(infer_tone_from_row, axis=1)
    else:
        missing_tone = df["emotional_tone"].isna() | (df["emotional_tone"].astype(str).str.strip() == "")
        df.loc[missing_tone, "emotional_tone"] = df[missing_tone].apply(infer_tone_from_row, axis=1)
    return df


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
            max_features=5000,
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def train_tone_classifier(df, test_size=0.25):
    work = df.dropna(subset=["title_clean", "emotional_tone"]).copy()
    work = work[work["title_clean"].str.len() > 0]
    stratify = work["emotional_tone"] if work["emotional_tone"].value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        work["title_clean"],
        work["emotional_tone"],
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    labels = sorted(work["emotional_tone"].unique())
    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, labels=labels, zero_division=0, output_dict=True),
        "matrix": confusion_matrix(y_test, predictions, labels=labels),
        "labels": labels,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
    }


def save_model(model, path=MODEL_FILE):
    joblib.dump(model, path)


def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def log_metrics(result, path=TRACKING_FILE):
    metrics = {
        "logged_at": utc_now(),
        "backend": "mlflow" if dependency_status()["mlflow"] else "json_fallback",
        "accuracy": result["accuracy"],
        "labels": result["labels"],
        "train_rows": len(result["x_train"]),
        "test_rows": len(result["x_test"]),
        "model_file": MODEL_FILE,
    }

    if dependency_status()["mlflow"]:
        import mlflow

        mlflow.set_experiment("FeedBalance_Phase5_EmotionalTone")
        with mlflow.start_run():
            mlflow.log_param("model", "TfidfVectorizer + LogisticRegression")
            mlflow.log_param("labels", ",".join(result["labels"]))
            mlflow.log_metric("accuracy", result["accuracy"])
            if os.path.exists(MODEL_FILE):
                mlflow.log_artifact(MODEL_FILE)
        metrics["backend"] = "mlflow"

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def ensure_db_schema(db_path=DB_FILE):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tone_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content_id TEXT NOT NULL,
                title TEXT,
                predicted_tone TEXT NOT NULL,
                confidence REAL,
                generated_at TEXT NOT NULL
            )
        """)


def update_database_tones(df, model, db_path=DB_FILE, user_id=DEFAULT_USER_ID):
    if not os.path.exists(db_path):
        return {"updated_watch_rows": 0, "tone_events": 0}

    ensure_db_schema(db_path)
    work = df.copy()
    work["predicted_tone"] = model.predict(work["title_clean"])

    confidences = []
    if hasattr(model.named_steps["model"], "predict_proba"):
        probabilities = model.predict_proba(work["title_clean"])
        confidences = probabilities.max(axis=1).tolist()
    else:
        confidences = [None] * len(work)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM tone_events WHERE user_id = ?", (user_id,))
        updated = 0
        for (_, row), confidence in zip(work.iterrows(), confidences):
            cursor = conn.execute(
                """
                UPDATE watch_history
                SET emotional_tone = ?
                WHERE user_id = ? AND content_id = ?
                """,
                (row["predicted_tone"], user_id, str(row["content_id"])),
            )
            updated += cursor.rowcount
            conn.execute(
                """
                INSERT INTO tone_events (
                    user_id, content_id, title, predicted_tone, confidence, generated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    str(row["content_id"]),
                    str(row["title"]),
                    row["predicted_tone"],
                    None if confidence is None else float(confidence),
                    utc_now(),
                ),
            )
    return {"updated_watch_rows": updated, "tone_events": len(work)}


def tone_distribution_from_db(db_path=DB_FILE, user_id=DEFAULT_USER_ID):
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["emotional_tone", "rows"])
    with sqlite3.connect(db_path) as conn:
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
        page_title="FeedBalance Phase 5",
        page_icon="NLP",
        layout="wide",
    )

    st.title("FeedBalance - Phase 5")
    st.caption("Emotional-tone classifier for mental health-aware feed analysis.")

    with st.sidebar:
        st.title("FeedBalance")
        st.caption("Phase 5 - AI / NLP")
        st.divider()
        dataset_path = st.text_input("Dataset path", DATASET_FILE)
        user_id = st.text_input("User ID", DEFAULT_USER_ID)
        test_size = st.slider("Test size", 0.15, 0.40, 0.25, 0.05)
        st.divider()
        st.code("streamlit run feedbalance_phase5.py", language="bash")

    try:
        df = load_dataset(dataset_path)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if df.empty:
        st.error("No dataset found. Complete Phase 2 first.")
        st.stop()

    result = train_tone_classifier(df, test_size=test_size)
    save_model(result["pipeline"])
    metrics = log_metrics(result)
    db_update = update_database_tones(df, result["pipeline"], user_id=user_id)
    db_dist = tone_distribution_from_db(user_id=user_id)
    deps = dependency_status()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows", len(df))
    metric_cols[1].metric("Tone Labels", df["emotional_tone"].nunique())
    metric_cols[2].metric("Accuracy", f"{result['accuracy']:.1%}")
    metric_cols[3].metric("DB Rows Updated", db_update["updated_watch_rows"])
    metric_cols[4].metric("Tracking", metrics["backend"])

    tab_model, tab_predict, tab_db, tab_learn = st.tabs([
        "Model",
        "Live Prediction",
        "Database",
        "Learn",
    ])

    with tab_model:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Dependency status")
            st.json(deps)
            st.subheader("Tone balance")
            st.dataframe(df["emotional_tone"].value_counts().rename("rows"), use_container_width=True)
        with right:
            st.subheader("Classification report")
            st.dataframe(pd.DataFrame(result["report"]).transpose().round(3), use_container_width=True)

        st.subheader("Confusion matrix")
        matrix = pd.DataFrame(result["matrix"], index=result["labels"], columns=result["labels"])
        st.dataframe(matrix, use_container_width=True)

    with tab_predict:
        st.subheader("Live emotional-tone prediction")
        model = load_model() or result["pipeline"]
        text = st.text_input("Content title", "Discipline over motivation")
        if text:
            prediction = model.predict([clean_text(text)])[0]
            st.metric("Predicted Tone", prediction)
            if hasattr(model.named_steps["model"], "predict_proba"):
                probs = model.predict_proba([clean_text(text)])[0]
                prob_df = pd.DataFrame({
                    "tone": model.classes_,
                    "confidence": probs,
                }).sort_values("confidence", ascending=False)
                st.dataframe(prob_df.round(3), use_container_width=True)

    with tab_db:
        st.subheader("Tone distribution stored in SQLite")
        st.dataframe(db_dist, use_container_width=True)
        st.subheader("Tracking artifact")
        st.json(metrics)

    with tab_learn:
        st.subheader("Phase 5 concepts")
        st.markdown("""
        | Component | Implementation |
        |---|---|
        | Emotional tone labels | Aspirational, Humorous, Melancholic, Outrage, Educational, Romantic, Relaxing |
        | NLP classifier | Local TF-IDF + Logistic Regression model, saved as `emotional_tone_classifier.pkl` |
        | HuggingFace bridge | Dependency check shows when `torch` and `transformers` are available for BERT upgrade |
        | MLOps tracking | Uses MLflow when installed, otherwise writes `phase5_tone_metrics.json` |
        | Database integration | Updates `watch_history.emotional_tone` and records `tone_events` |
        | Phase 6 bridge | Tone-aware analytics are ready for Streamlit Cloud deployment |
        """)

    st.divider()
    st.caption("FeedBalance Phase 5 - Emotional Tone NLP - BERT-ready architecture - MLOps tracking")


if __name__ == "__main__":
    main()
