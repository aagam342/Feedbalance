# FeedBalance

FeedBalance is a mental health-aware content diversity system. It detects category saturation, recommends balancing content, and tracks emotional tone for feed wellbeing.

## Current Build

- Phase 1: Python/OOP saturation detector
- Phase 2: Data science dashboard and CSV dataset
- Phase 3: Category classifier
- Phase 4: SQLite recommendation engine
- Phase 5: Emotional-tone NLP layer
- Phase 6: Deployment-ready Streamlit app

## Run

```bash
streamlit run app.py
```

## Main Files

- `app.py` - production Streamlit entrypoint
- `feedbalance_phase1.py` to `feedbalance_phase5.py` - phase apps and implementation
- `feedbalance_dataset.csv` - training/content dataset
- `category_classifier.pkl` - category ML model
- `emotional_tone_classifier.pkl` - tone ML model
- `feedbalance.db` - SQLite demo database

## Deploy

Deploy `app.py` on Streamlit Community Cloud with `requirements.txt`.
