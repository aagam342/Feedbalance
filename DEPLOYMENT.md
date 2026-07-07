# FeedBalance Phase 6 Deployment

## Local Run

```bash
cd D:\Feedbalance
streamlit run app.py
```

## Streamlit Community Cloud

Use `app.py` as the app entrypoint.

Required repo files:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `feedbalance_dataset.csv`
- `category_classifier.pkl`
- `emotional_tone_classifier.pkl`
- `feedbalance.db`

Optional secret:

```toml
YOUTUBE_API_KEY = "your_key_here"
```

Set secrets in Streamlit Community Cloud advanced settings. Do not commit `.env` or `.streamlit/secrets.toml`.

## Deployment Checklist

- Push this folder to GitHub.
- Create a Streamlit Community Cloud app.
- Select the repository, branch, and `app.py`.
- Paste secrets if YouTube API access is needed.
- Deploy and confirm the Overview, Recommendations, Predict, and Data tabs render.
