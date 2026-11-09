# app.py
import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


if not (os.path.exists("student_model.pkl") and os.path.exists("student_data.csv")):
    st.error("Model or data file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
data = pd.read_csv("student_data.csv")

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration. "
           "Predictions are not diagnoses and must not be the sole basis "
           "for real student decisions.")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

# Model evaluation summary
if os.path.exists("metrics.json"):
    with open("metrics.json") as f:
        m = json.load(f)
    st.subheader("Model evaluation summary (test set)")
    rep = m["report"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Accuracy", round(m["accuracy"], 3))
    e2.metric("At-risk precision", round(rep["At risk"]["precision"], 3))
    e3.metric("At-risk recall", round(rep["At risk"]["recall"], 3))
    e4.metric("At-risk F1", round(rep["At risk"]["f1-score"], 3))

st.subheader("Dataset preview")
st.dataframe(data.head(10), width="stretch")

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file as CSV: {exc}")
        st.stop()

    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK" for value in predictions
        ]

        st.subheader("Prediction results")
        st.dataframe(results, width="stretch")
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))

        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv",
        )
