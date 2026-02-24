import json
import os
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(page_title="PowerCast AI", page_icon="🔋", layout="wide")

st.title("🔋 PowerCast AI")
st.subheader("Energy Consumption Forecasting Dashboard")

st.markdown(
    """
    Welcome to **PowerCast AI** — an AI-powered energy consumption forecasting tool.
    Upload your data, view predictions, and share your feedback to help us improve.
    """
)

st.divider()

# ── Forecast Placeholder ─────────────────────────────────────────────────────
st.header("📊 Forecast Overview")
st.info(
    "Forecast visualizations will appear here once a dataset is loaded. "
    "Please follow the instructions in the README to download and place the "
    "dataset in the `data/` folder, then restart the app."
)

st.divider()

# ── User Feedback ─────────────────────────────────────────────────────────────
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback.json")


def load_feedback() -> list:
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_feedback(entries: list) -> None:
    with open(FEEDBACK_FILE, "w") as fh:
        json.dump(entries, fh, indent=2)


st.header("💬 User Feedback")
st.markdown(
    "We value your input! Please rate the forecast quality and share any thoughts "
    "to help us make PowerCast AI better."
)

with st.form("feedback_form", clear_on_submit=True):
    name = st.text_input("Your name (optional)")
    rating = st.slider(
        "Rate the forecast quality",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Poor, 5 = Excellent",
    )
    comment = st.text_area("Additional comments (optional)", max_chars=500)
    submitted = st.form_submit_button("Submit Feedback")

if submitted:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name.strip() if name else "Anonymous",
        "rating": rating,
        "comment": comment.strip(),
    }
    entries = load_feedback()
    entries.append(entry)
    save_feedback(entries)
    st.success("✅ Thank you for your feedback!")

# ── Display existing feedback ─────────────────────────────────────────────────
entries = load_feedback()
if entries:
    st.subheader("📝 Recent Feedback")
    for entry in reversed(entries[-10:]):
        stars = "⭐" * entry["rating"]
        ts = entry["timestamp"].split("T")[0]
        with st.expander(f"{stars}  —  {entry['name']}  ({ts})"):
            if entry["comment"]:
                st.write(entry["comment"])
            else:
                st.write("_No additional comment._")
