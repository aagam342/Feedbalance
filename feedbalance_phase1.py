# ─────────────────────────────────────────────────────────────────
# FEEDBALANCE — PHASE 1
# Fake Dataset Generator + Category Saturation Detector
#
# RUN: streamlit run feedbalance_phase1.py
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import random
import json
import os
import math
from datetime import datetime, timedelta

# ── CONFIG — must be FIRST streamlit call ─────────────────────────
st.set_page_config(
    page_title="FeedBalance Phase 1",
    page_icon="📡",
    layout="wide"
)

# ── CONSTANTS ─────────────────────────────────────────────────────
CATEGORIES = [
    "fitness", "food", "comedy", "love", "news",
    "motivation", "quotes", "cricket", "tech", "music"
]

CAT_EMOJI = {
    "fitness":"💪","food":"🍕","comedy":"😂","love":"❤️",
    "news":"📰","motivation":"🔥","quotes":"💭",
    "cricket":"🏏","tech":"💻","music":"🎵"
}

CAT_COLOR = {
    "fitness":"#ef4444","food":"#f97316","comedy":"#eab308",
    "love":"#ec4899","news":"#6366f1","motivation":"#f59e0b",
    "quotes":"#8b5cf6","cricket":"#10b981","tech":"#3b82f6",
    "music":"#06b6d4"
}

SAMPLE_TITLES = {
    "fitness":   ["5 Best Morning Workouts","Chest Day Routine",
                  "Lose Weight in 30 Days","Home Gym Setup","Protein Diet Plan"],
    "food":      ["Pasta Recipe in 10 Min","Street Food Tour Mumbai",
                  "Biryani Secrets","Healthy Breakfast Ideas","5-Ingredient Desserts"],
    "comedy":    ["Office Life Be Like","Mom vs Kids",
                  "When WiFi Stops Working","Exam Season Mood","Autowaala Stories"],
    "love":      ["Propose Karo Aise","Long Distance Tips",
                  "First Date Red Flags","Couple Goals 2026","True Love Signs"],
    "news":      ["Breaking: Market Update","Political Drama Today",
                  "Weather Alert India","World News Roundup","Economy Report"],
    "motivation":["Wake Up at 5AM","Your Dream Won't Wait",
                  "No Excuses Mode","Level Up Daily","Discipline Over Motivation"],
    "quotes":    ["Life Is Too Short","Find Your Purpose",
                  "Silence Is Power","Trust The Process","Be The Change"],
    "cricket":   ["IPL 2026 Highlights","Virat's Best Shots",
                  "India vs Pakistan Recap","Bowling Masterclass","Fantasy Team Tips"],
    "tech":      ["AI Will Change Everything","Python in 10 Minutes",
                  "Best Gadgets 2026","Coding Tips for Freshers","ChatGPT vs Claude"],
    "music":     ["Lofi Beats to Study","Top Bollywood 2026",
                  "Guitar for Beginners","Music Therapy Works","New Album Review"],
}

SAVE_FILE = "watch_history.json"


# ════════════════════════════════════════════════════════════════
# CORE FUNCTIONS  (Week 1-4 concepts)
# ════════════════════════════════════════════════════════════════

# WEEK 1 — Variables + If-Else
def is_saturated(share: float, threshold: float) -> bool:
    if share > threshold:
        return True
    else:
        return False

# WEEK 2 — Lists + Dicts + Functions
def generate_fake_history(n: int, bias_category=None, bias_strength=0.60) -> list:
    history = []
    base_time = datetime.now() - timedelta(days=7)
    for i in range(n):
        if bias_category and random.random() < bias_strength:
            category = bias_category
        else:
            category = random.choice(CATEGORIES)
        event = {
            "id":         i + 1,
            "category":   category,
            "title":      random.choice(SAMPLE_TITLES[category]),
            "watch_time": random.randint(15, 180),
            "liked":      random.random() > 0.55,
            "shared":     random.random() > 0.80,
            "timestamp":  (base_time + timedelta(
                               minutes=random.randint(0, 7*24*60)
                           )).strftime("%d %b %Y, %I:%M %p"),
        }
        history.append(event)
    return history

def count_categories(history: list) -> dict:
    counts = {}
    for event in history:
        cat = event["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts

def calculate_shares(counts: dict, total: int) -> dict:
    if total == 0:
        return {}
    return {cat: round(count / total, 4) for cat, count in counts.items()}

def get_top_category(shares: dict):
    if not shares:
        return None, 0.0
    top_cat = max(shares, key=shares.get)
    return top_cat, shares[top_cat]

# WEEK 3 — JSON + File I/O
def save_history(history: list, filename=SAVE_FILE) -> bool:
    try:
        with open(filename, "w") as f:
            json.dump(history, f, indent=2)
        return True
    except IOError:
        return False

def load_history(filename=SAVE_FILE) -> list:
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

# WEEK 4 — OOP Classes
class WatchHistory:
    def __init__(self, max_window: int = 50):
        self.items = []
        self.max_window = max_window

    def add_item(self, event: dict):
        self.items.append(event)
        if len(self.items) > self.max_window:
            self.items.pop(0)

    def load_from_list(self, data: list):
        self.items = data[-self.max_window:]

    def get_counts(self) -> dict:
        return count_categories(self.items)

    def get_shares(self) -> dict:
        return calculate_shares(self.get_counts(), len(self.items))

    def save(self) -> bool:
        return save_history(self.items)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.items)


class SaturationDetector:
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def analyze(self, wh: WatchHistory) -> dict:
        shares       = wh.get_shares()
        top_cat, top_share = get_top_category(shares)
        saturated    = is_saturated(top_share, self.threshold)
        diversity    = self._shannon_entropy(shares)
        return {
            "saturated":       saturated,
            "top_category":    top_cat,
            "top_share":       top_share,
            "diversity_score": diversity,
            "shares":          shares,
            "total":           len(wh.items),
        }

    def _shannon_entropy(self, shares: dict) -> float:
        active = [p for p in shares.values() if p > 0]
        if len(active) <= 1:
            return 0.0
        entropy     = -sum(p * math.log2(p) for p in active)
        max_entropy = math.log2(len(active))
        return round(entropy / max_entropy, 3) if max_entropy > 0 else 0.0


# ════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ════════════════════════════════════════════════════════════════
if "wh" not in st.session_state:
    st.session_state.wh = WatchHistory()
    for ev in generate_fake_history(30):
        st.session_state.wh.add_item(ev)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📡 FeedBalance")
    st.caption("Phase 1 — Saturation Detector")
    st.divider()

    st.subheader("⚙️ Controls")

    n_items = st.slider("History Size", 10, 100, 30, 5)

    bias_options = ["None (Random)"] + CATEGORIES
    bias_sel = st.selectbox(
        "Bias Category (60% skew)",
        bias_options,
        format_func=lambda x: f"{CAT_EMOJI.get(x,'')} {x}"
                               if x != "None (Random)" else "🎲 None (Random)"
    )
    bias_cat = None if bias_sel == "None (Random)" else bias_sel

    threshold = st.slider("Saturation Threshold", 0.20, 0.70, 0.40, 0.05,
                          format="%.0f%%")

    if st.button("⚡ Generate Dataset", use_container_width=True):
        wh_new = WatchHistory(max_window=n_items)
        for ev in generate_fake_history(n_items, bias_cat):
            wh_new.add_item(ev)
        st.session_state.wh = wh_new
        st.success(f"✅ {n_items} events generated!")
        st.rerun()

    st.divider()
    st.subheader("💾 Persistence")

    col_s, col_l = st.columns(2)
    with col_s:
        if st.button("Save JSON", use_container_width=True):
            if st.session_state.wh.save():
                st.success("Saved!")
            else:
                st.error("Save failed")
    with col_l:
        if st.button("Load JSON", use_container_width=True):
            data = load_history()
            if data:
                wh_loaded = WatchHistory()
                wh_loaded.load_from_list(data)
                st.session_state.wh = wh_loaded
                st.success(f"Loaded {len(data)} items!")
                st.rerun()
            else:
                st.warning("No saved file found")

    st.divider()
    st.caption("""
    **Week 1** — Variables, If-Else  
    **Week 2** — Lists, Dicts, Functions  
    **Week 3** — JSON, File I/O  
    **Week 4** — Classes, OOP  
    """)


# ════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════════
wh       = st.session_state.wh
detector = SaturationDetector(threshold=threshold)
result   = detector.analyze(wh)

st.title("📡 FeedBalance — Phase 1")
st.caption("Fake Dataset Generator + Category Saturation Detector")
st.divider()

# ── TOP METRICS ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

ds = result["diversity_score"]
ds_delta = "🌿 Healthy" if ds >= 0.7 else "⚡ Warning" if ds >= 0.4 else "🔴 Critical"

c1.metric("Total Events",     result["total"])
c2.metric("Diversity Score",  f"{ds:.2f} / 1.0",  ds_delta)
c3.metric("Top Category",
          f"{CAT_EMOJI.get(result['top_category'],'')} {result['top_category']}",
          f"{result['top_share']:.0%} share")
c4.metric("Feed Status",
          "⚠️ SATURATED" if result["saturated"] else "✅ HEALTHY")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Dashboard",
    "📋 Watch History",
    "📚 Learn Concepts",
    "💻 Full Code"
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("📊 Category Distribution")

        shares_sorted = sorted(
            result["shares"].items(), key=lambda x: x[1], reverse=True
        )

        for cat, share in shares_sorted:
            pct        = share * 100
            sat_flag   = share >= threshold
            color      = CAT_COLOR.get(cat, "#64748b")
            emoji      = CAT_EMOJI.get(cat, "")

            label = f"{emoji} **{cat}**"
            if sat_flag:
                label += "  🔴 *SATURATED*"

            st.markdown(label)
            st.progress(share, text=f"{pct:.0f}%")

    with right:
        st.subheader("🌿 Status")

        # Diversity gauge via progress bar
        ds_pct = int(ds * 100)
        ds_label = "🌿 Healthy" if ds_pct >= 70 else "⚡ Warning" if ds_pct >= 40 else "🔴 Critical"
        st.markdown(f"**Diversity Score: {ds_pct} / 100**")
        st.progress(ds, text=ds_label)

        st.divider()

        # Alert
        if result["saturated"]:
            st.error(
                f"⚠️ **SATURATION DETECTED!**\n\n"
                f"{CAT_EMOJI.get(result['top_category'],'')} "
                f"**{result['top_category'].upper()}** is "
                f"{result['top_share']:.0%} of your feed.\n\n"
                f"Limit: {threshold:.0%} → Diversity needed!"
            )
        else:
            st.success(
                f"✅ **Feed is Healthy!**\n\n"
                f"Good variety — no category above {threshold:.0%}."
            )

        st.divider()

        # Quick stats
        st.subheader("📈 Quick Stats")
        liked   = sum(1 for i in wh.items if i.get("liked"))
        avg_wt  = sum(i["watch_time"] for i in wh.items) / max(len(wh.items), 1)

        st.markdown(f"""
        | Stat | Value |
        |------|-------|
        | Categories Active | {len(result['shares'])} |
        | Total Liked | {liked} / {result['total']} |
        | Avg Watch Time | {avg_wt:.0f} sec |
        | Threshold | {threshold:.0%} |
        """)


# ════════════════════════════════════════════════════════════════
# TAB 2 — WATCH HISTORY
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📋 Generated Watch History")
    st.caption(f"{len(wh.items)} events total")

    if wh.items:
        df = wh.to_dataframe()

        # Emoji column
        df.insert(0, "  ", df["category"].map(CAT_EMOJI))

        st.dataframe(
            df,
            use_container_width=True,
            height=450,
            hide_index=True,
            column_config={
                "liked":  st.column_config.CheckboxColumn("Liked"),
                "shared": st.column_config.CheckboxColumn("Shared"),
            }
        )

        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "feedbalance_history.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_json:
            st.download_button(
                "⬇️ Download JSON",
                json.dumps(wh.items, indent=2).encode("utf-8"),
                "watch_history.json",
                "application/json",
                use_container_width=True,
            )


# ════════════════════════════════════════════════════════════════
# TAB 3 — LEARN CONCEPTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📚 Week-by-Week Concepts Used in This App")

    with st.expander("**Week 1 — Variables & If-Else**", expanded=False):
        st.info("**Theory:** Variable ek container hai. If-Else program ko decisions lene deta hai. FeedBalance mein: agar category share 40% cross kare → saturation alert.")
        st.success("**FeedBalance Use:** `is_saturated()` function — sirf ek if-else. Yahi tera pehla detection logic hai.")
        st.code('''# Variables
category   = "fitness"   # str
watch_time = 45          # int
share      = 0.55        # float
liked      = True        # bool

# If-Else — saturation check
threshold = 0.40

if share > threshold:
    print("⚠️ SATURATION DETECTED!")
elif share > 0.30:
    print("⚡ Warning: Getting high")
else:
    print("✅ Feed is healthy")''', language="python")

    with st.expander("**Week 2 — Lists, Dictionaries, Functions**", expanded=False):
        st.info("**Theory:** List = ordered collection. Dictionary = key-value store. Function = reusable code block.")
        st.success("**FeedBalance Use:** `generate_fake_history()` list of dicts return karta hai. `count_categories()` har category ka count dict mein rakhta hai.")
        st.code('''# List of Dictionaries — watch history
history = []

def generate_event(category):
    return {
        "category":   category,     # str
        "watch_time": 45,           # int
        "liked":      True          # bool
    }

for i in range(5):
    event = generate_event("fitness")
    history.append(event)

# Count categories
def count_categories(history):
    counts = {}
    for event in history:
        cat = event["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts

print(count_categories(history))
# {"fitness": 5}''', language="python")

    with st.expander("**Week 3 — JSON & File I/O**", expanded=False):
        st.info("**Theory:** JSON = file format jisme Python data save hota hai. Program band karo — data survive karta hai. Yeh PERSISTENCE hai.")
        st.success("**FeedBalance Use:** Sidebar mein 'Save JSON' button `save_history()` call karta hai. 'Load JSON' button `load_history()` call karta hai.")
        st.code('''import json
import os

def save_history(history, filename="watch_history.json"):
    try:
        with open(filename, "w") as f:
            json.dump(history, f, indent=2)
        print(f"✅ Saved {len(history)} items")
    except IOError as e:
        print(f"❌ Error: {e}")

def load_history(filename="watch_history.json"):
    if not os.path.exists(filename):
        return []               # File nahi mili
    with open(filename, "r") as f:
        return json.load(f)

# Use
save_history(history)
loaded = load_history()
print(f"Loaded: {len(loaded)} items")''', language="python")

    with st.expander("**Week 4 — OOP: Classes & Objects**", expanded=False):
        st.info("**Theory:** Class = blueprint. Object = us blueprint ka product. Related data + functions ek jagah wrap hoti hain.")
        st.success("**FeedBalance Use:** `WatchHistory` class aur `SaturationDetector` class — yeh app ke do main building blocks hain.")
        st.code('''class WatchHistory:
    def __init__(self, max_window=30):  # Constructor
        self.items      = []            # Instance variable
        self.max_window = max_window

    def add_item(self, event):          # Method
        self.items.append(event)
        if len(self.items) > self.max_window:
            self.items.pop(0)           # Rolling window

    def get_counts(self):
        counts = {}
        for item in self.items:
            c = item["category"]
            counts[c] = counts.get(c, 0) + 1
        return counts

class SaturationDetector:
    def __init__(self, threshold=0.40):
        self.threshold = threshold

    def analyze(self, history_obj):     # Doosra object leta hai
        shares = history_obj.get_shares()
        top    = max(shares, key=shares.get)
        return {
            "saturated": shares[top] > self.threshold,
            "top":       top,
            "share":     shares[top]
        }

# Use karo
wh  = WatchHistory()
wh.add_item({"category": "fitness", "watch_time": 45})
det = SaturationDetector(threshold=0.40)
print(det.analyze(wh))''', language="python")

    with st.expander("**Shannon Entropy — Diversity Score Formula**", expanded=False):
        st.info("**Theory:** Shannon Entropy information theory ka concept hai. Score 0.0 to 1.0 — 1.0 = perfectly diverse, 0.0 = ek hi category.")
        st.success("**FeedBalance Use:** `SaturationDetector._shannon_entropy()` — yeh diversity meter ko power karta hai.")
        st.code('''import math

def diversity_score(shares):
    """
    H = -sum(p * log2(p))
    Normalized: score = H / log2(n)
    """
    active = [p for p in shares.values() if p > 0]
    if len(active) <= 1:
        return 0.0

    entropy     = -sum(p * math.log2(p) for p in active)
    max_entropy = math.log2(len(active))
    return entropy / max_entropy

# Low diversity (one category dominates)
low  = {"fitness": 0.80, "food": 0.20}
print(diversity_score(low))    # ~0.72

# High diversity (equal spread across 5)
high = {"fitness":0.2, "food":0.2, "comedy":0.2,
        "love":0.2, "news":0.2}
print(diversity_score(high))   # 1.0''', language="python")


# ════════════════════════════════════════════════════════════════
# TAB 4 — FULL CODE
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💻 Setup Guide")

    st.markdown("#### Step 1 — Install dependencies")
    st.code("pip install streamlit pandas", language="bash")

    st.markdown("#### Step 2 — Run the app")
    st.code("streamlit run feedbalance_phase1.py", language="bash")

    st.markdown("#### Step 3 — Project folder structure")
    st.code("""feedbalance/
├── feedbalance_phase1.py    ← Yeh file
├── watch_history.json       ← Auto-create hogi (Save button se)
└── README.md                ← GitHub ke liye (baad mein)""",
            language="text")

    st.divider()
    st.subheader("📦 What's Next")

    st.markdown("""
    | Phase | Topics | FeedBalance Feature |
    |-------|--------|---------------------|
    | ✅ Phase 1 (Now) | Python, OOP, JSON | Fake dataset + Saturation detector |
    | 🔜 Phase 2 | NumPy, Pandas, Matplotlib | Real data analysis + Charts |
    | 🔜 Phase 3 | Scikit-learn ML | Content category classifier (85%+ acc) |
    | 🔜 Phase 4 | Recommendation Systems | Diversity-aware engine |
    | 🔜 Phase 5 | HuggingFace, BERT | Emotional tone classifier |
    | 🔜 Phase 6 | Deployment | Live on streamlit.io |
    """)


# ── FOOTER ────────────────────────────────────────────────────────
st.divider()
st.caption("FeedBalance Phase 1 · Python Weeks 1–4 · Variables · Lists · Dicts · Functions · JSON · OOP · 2026")