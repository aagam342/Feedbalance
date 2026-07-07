"""FeedBalance Phase 3: content category classifier."""

import os
from collections import Counter

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


DATASET_FILE = "feedbalance_dataset.csv"
MODEL_FILE = "category_classifier.pkl"
MIN_CLASS_ROWS_FOR_SPLIT = 2


def clean_text(value):
    return str(value).strip().lower()


def load_dataset(path=DATASET_FILE):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["title", "category"])

    df = pd.read_csv(path)
    required = {"title", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["title", "category"]).copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df = df[df["title_clean"].str.len() > 0]
    return df


def build_pipeline(c=1.0, max_features=5000):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            strip_accents="unicode",
        )),
        ("model", LogisticRegression(
            C=c,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def can_stratify(y):
    counts = Counter(y)
    return len(counts) > 1 and min(counts.values()) >= MIN_CLASS_ROWS_FOR_SPLIT


def train_baseline(df, test_size=0.25):
    x = df["title_clean"]
    y = df["category"]
    stratify = y if can_stratify(y) else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    labels = sorted(y.unique())
    report = classification_report(y_test, predictions, labels=labels, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return {
        "pipeline": pipeline,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "labels": labels,
        "accuracy": accuracy_score(y_test, predictions),
        "report": report,
        "matrix": matrix,
    }


def tune_model(df):
    x = df["title_clean"]
    y = df["category"]
    min_class_count = min(Counter(y).values())
    splits = max(2, min(5, min_class_count))

    pipeline = build_pipeline()
    params = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [1000, 3000, 5000],
        "model__C": [0.5, 1.0, 2.0, 5.0],
    }
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        pipeline,
        params,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        error_score=0,
    )
    search.fit(x, y)
    return search


def save_model(pipeline, path=MODEL_FILE):
    joblib.dump(pipeline, path)


def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def cached_dataset(path):
    return load_dataset(path)


def cached_training(path, test_size):
    df = load_dataset(path)
    if len(df) < 10 or df["category"].nunique() < 2:
        return None
    return train_baseline(df, test_size=test_size)


def main():
    st.set_page_config(
        page_title="FeedBalance Phase 3",
        page_icon="ML",
        layout="wide",
    )

    st.title("FeedBalance - Phase 3")
    st.caption("Scikit-learn content category classifier using TF-IDF and Logistic Regression.")
    
    with st.sidebar:
        st.title("FeedBalance")
        st.caption("Phase 3 - Machine Learning")
        st.divider()
        dataset_path = st.text_input("Dataset path", DATASET_FILE)
        test_size = st.slider("Test size", 0.15, 0.40, 0.25, 0.05)
        auto_save = st.checkbox("Save baseline model after training", value=True)
        run_tuning = st.checkbox("Run GridSearchCV tuning", value=False)
        st.divider()
        st.code("streamlit run feedbalance_phase3.py", language="bash")
    
    
    try:
        df = cached_dataset(dataset_path)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    
    if df.empty:
        st.error("No dataset found. Complete Phase 2 first so feedbalance_dataset.csv exists.")
        st.stop()
    
    category_counts = df["category"].value_counts()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", len(df))
    metric_cols[1].metric("Categories", df["category"].nunique())
    metric_cols[2].metric("Largest Class", category_counts.index[0], int(category_counts.iloc[0]))
    metric_cols[3].metric("Model File", "Ready" if os.path.exists(MODEL_FILE) else "Not saved")
    
    result = cached_training(dataset_path, test_size)
    if result is None:
        st.error("Need at least 10 rows and 2 categories to train a classifier.")
        st.stop()
    
    if auto_save:
        save_model(result["pipeline"])
    
    tab_train, tab_tune, tab_predict, tab_data, tab_learn = st.tabs([
        "Training",
        "Tuning",
        "Live Prediction",
        "Dataset",
        "Learn",
    ])
    
    with tab_train:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Baseline model")
            accuracy = result["accuracy"]
            delta = "Target hit" if accuracy >= 0.75 else "Needs more data"
            st.metric("Accuracy", f"{accuracy:.1%}", delta)
            st.write("Pipeline")
            st.code("TfidfVectorizer(ngram_range=(1, 2)) + LogisticRegression", language="text")
            st.write("Train/Test split")
            st.write(f"{len(result['x_train'])} training titles, {len(result['x_test'])} test titles")
            if os.path.exists(MODEL_FILE):
                st.success(f"Saved model: {MODEL_FILE}")
    
        with right:
            st.subheader("Classification report")
            report_df = pd.DataFrame(result["report"]).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
    
        st.subheader("Confusion matrix")
        matrix_df = pd.DataFrame(result["matrix"], index=result["labels"], columns=result["labels"])
        st.dataframe(matrix_df, use_container_width=True)
    
    with tab_tune:
        st.subheader("GridSearchCV")
        if run_tuning:
            with st.spinner("Running hyperparameter search..."):
                search = tune_model(df)
            st.metric("Best CV Accuracy", f"{search.best_score_:.1%}")
            st.write("Best parameters")
            st.json(search.best_params_)
            if st.button("Save tuned model", use_container_width=True):
                save_model(search.best_estimator_)
                st.success(f"Saved tuned model to {MODEL_FILE}")
        else:
            st.info("Enable GridSearchCV tuning in the sidebar to run a compact hyperparameter search.")
    
    with tab_predict:
        st.subheader("Live category prediction")
        model = load_model() or result["pipeline"]
        sample_titles = [
            "Best Python tips for beginners",
            "IPL final match highlights",
            "Healthy breakfast recipe in 10 minutes",
            "Morning workout routine for weight loss",
        ]
        title = st.text_input("Content title", sample_titles[0])
        if title:
            prediction = model.predict([clean_text(title)])[0]
            probabilities = None
            if hasattr(model.named_steps["model"], "predict_proba"):
                probabilities = model.predict_proba([clean_text(title)])[0]
            st.metric("Predicted Category", prediction)
            if probabilities is not None:
                prob_df = pd.DataFrame({
                    "category": model.classes_,
                    "confidence": probabilities,
                }).sort_values("confidence", ascending=False)
                st.dataframe(prob_df.round(3), use_container_width=True)
        st.write("Try one")
        st.write(", ".join(sample_titles))
    
    with tab_data:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Class balance")
            st.dataframe(category_counts.rename("rows"), use_container_width=True)
        with right:
            st.subheader("Training rows")
            st.dataframe(df[["title", "title_clean", "category", "source"]].head(100), use_container_width=True, height=460)
    
    with tab_learn:
        st.subheader("Phase 3 concepts")
        st.markdown("""
        | Step | FeedBalance implementation |
        |---|---|
        | Load Phase 2 data | Reads `feedbalance_dataset.csv` |
        | Preprocess text | Lowercases and strips `title` into `title_clean` |
        | Feature extraction | `TfidfVectorizer(ngram_range=(1, 2))` |
        | Classifier | `LogisticRegression(class_weight="balanced")` |
        | Evaluation | `train_test_split`, accuracy, classification report, confusion matrix |
        | Tuning | `GridSearchCV` over n-grams, max features, and regularization |
        | Persistence | `joblib.dump(..., "category_classifier.pkl")` |
        """)
    
    st.divider()
    st.caption("FeedBalance Phase 3 - Scikit-learn - TF-IDF - Logistic Regression - joblib model export")

if __name__ == "__main__":
    main()
