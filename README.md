# Student Performance Prediction System

A classroom prototype that estimates whether a secondary-school student is at risk of failing the final Mathematics grade.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification using available student indicators (study time, absences, past failures, G1, G2, and more).

## Dataset
UCI Machine Learning Repository, Student Performance dataset: https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
`G3 < 10` = **AT RISK**; `G3 >= 10` = **NOT AT RISK**. G3 is the target only and is never used as an input feature.

## Technologies
Python, pandas, scikit-learn (Random Forest), Streamlit, joblib, GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, writes student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
Use "Download sample input CSV" in the app, then upload it to test predictions.

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions or populations.
- G1 and G2 are strong predictors of G3, so accuracy can look inflated; judge the model by at-risk recall, precision, and F1.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
