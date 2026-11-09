# train_model.py
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# Activity A: Retrieve and inspect the dataset
# ---------------------------------------------------------------
# Option 1: if the Mathematics file (student-mat.csv, from the UCI zip)
# is placed in this folder, use it (semicolon separated).
# Option 2: otherwise download through the ucimlrepo interface.
if os.path.exists("student-mat.csv"):
    df = pd.read_csv("student-mat.csv", sep=";")
    print("Loaded local student-mat.csv (Mathematics records)")
else:
    from ucimlrepo import fetch_ucirepo

    student = fetch_ucirepo(id=320)
    df = pd.concat([student.data.features, student.data.targets], axis=1)
    # Remove accidental duplicate column names, if any
    df = df.loc[:, ~df.columns.duplicated()]
    print("Downloaded dataset from UCI repository (id=320)")

# Checkpoint: G1, G2 and G3 must exist
for col in ["G1", "G2", "G3"]:
    assert col in df.columns, f"Column {col} is missing from the dataset"

df = df.drop_duplicates().reset_index(drop=True)
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# Activity B: Preprocess, train, evaluate
# ---------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy, 3))
report_text = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
)
print(report_text)

report_dict = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
    output_dict=True,
)
with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "report": report_dict}, f, indent=2)

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl and metrics.json")
