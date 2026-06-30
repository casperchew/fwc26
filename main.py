from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import LinearSVC

matches = pd.read_csv("data/WorldCupMatches.csv")
matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
matches = matches.drop(
    [
        "stadium",
        "city",
        "home_team_name",
        "away_team_name",
        "win_conditions",
        "attendance",
        "referee",
        "assistant_1",
        "assistant_2",
    ],
    axis=1,
)
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
matches = matches[matches.year > 2000]


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


X = matches[
    [
        "year",
        "month",
        "day",
        "half-time_home_goals",
        "half-time_away_goals",
        "home_team_initials",
        "away_team_initials",
    ]
]
y = matches.result

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

categorical_features = ["home_team_initials", "away_team_initials"]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            # OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

clf = Pipeline(
    [
        ("prep", preprocessor),
        ("model", OneVsOneClassifier(LinearSVC(random_state=0))),
    ]
)

clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test)}")

current_match = pd.DataFrame({
    "year": 2026,
    "month": 6,
    "day": 30,
    "half-time_home_goals": 0,
    "half-time_away_goals": 0,
    "home_team_initials": "NED",
    "away_team_initials": "MOR"
}, index=[0])

print(clf.predict(current_match))
