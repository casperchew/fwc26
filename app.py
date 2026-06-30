from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matches = pd.read_csv("data/WorldCupMatches.csv")
matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
matches = matches.dropna()
matches[
    [
        "year",
        "home_team_goals",
        "away_team_goals",
        "half-time_home_goals",
        "half-time_away_goals",
        "roundid",
        "matchid",
    ]
] = matches[
    [
        "year",
        "home_team_goals",
        "away_team_goals",
        "half-time_home_goals",
        "half-time_away_goals",
        "roundid",
        "matchid",
    ]
].astype(
    int
)


def datetime_converter(x):
    try:
        return datetime.strptime(x, "%d %b %Y - %H:%M ")
    except ValueError:
        return datetime.strptime(x, "%d %B %Y - %H:%M ")


matches.datetime = matches.datetime.apply(datetime_converter)
matches["month"] = matches.datetime.dt.month
matches["day"] = matches.datetime.dt.day
matches = matches.drop("datetime", axis=1)

matches.stage = matches.stage.replace(r"Group[\w\W]*", "Groups", regex=True)
matches.home_team_initials = matches.home_team_initials.astype("category")
matches.away_team_initials = matches.away_team_initials.astype("category")
matches["result"] = np.clip(matches.home_team_goals - matches.away_team_goals, -1, 1)

st.sidebar.write("Select columns:")
columns = {
    column: True
    for column in [
        "year",
        "half-time_home_goals",
        "half-time_away_goals",
        "home_team_initials",
        "away_team_initials",
        "month",
        "day",
    ]
}
for column in matches.columns:
    columns[column] = st.sidebar.checkbox(column, value=columns.get(column))

X = matches[[column for column in columns if columns[column]]]

categorical_features = [
    "stage",
    "stadium",
    "city",
    "home_team_name",
    "away_team_name",
    "win_conditions",
    "referee",
    "assistant_1",
    "assistant_2",
    "home_team_initials",
    "away_team_initials",
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(set(categorical_features) & set(X.columns)),
        )
    ],
    remainder="passthrough",
)

clf = Pipeline(
    [
        ("prep", preprocessor),
        ("model", LinearRegression()),
    ]
)


prediction_type = st.selectbox(label="Select prediction:", options=["Scoreline"])

y = matches[["home_team_goals", "away_team_goals"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf.fit(X_train, y_train)
st.write(f"Accuracy: {clf.score(X_test, y_test)}")

obs = {}

col1, col2 = st.columns(2)

with col1:
    obs["home_team_initials"] = st.selectbox(
        label="Home", options=sorted(matches.home_team_initials.unique())
    )

with col2:
    obs["away_team_initials"] = st.selectbox(
        label="Away", options=sorted(matches.home_team_initials.unique())
    )

now = datetime.now()
if columns["year"] or columns["month"] or columns["year"]:
    obs["selected_datetime"] = st.datetime_input("Date and time of match")

if st.button("Predict"):
    obs["year"] = obs["selected_datetime"].year
    obs["month"] = obs["selected_datetime"].month
    obs["day"] = obs["selected_datetime"].day
    pred = np.round(clf.predict(pd.DataFrame(obs, index=[0]))).astype(int)
    st.write(
        f"Predicted score: {obs['home_team_initials']} {pred[0][0]} - {pred[0][1]} {obs['away_team_initials']}"
    )
