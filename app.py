from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    Ridge,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import home_away_reverser, home_away_swap

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")


@st.cache_data
def load_matches():
    matches = pd.read_csv("data/matches_1930_2022.csv")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
    matches = matches[matches.year > 2000]
    matches = matches.drop(
        [
            "home_xg",
            "home_penalty",
            "away_xg",
            "away_penalty",
            "home_manager",
            "home_captain",
            "away_manager",
            "away_captain",
            "attendance",
            "venue",
            "officials",
            "score",
            "referee",
            "notes",
        ],
        axis=1,
    )
    matches.year -= 2026
    matches["result"] = np.clip(matches.home_score - matches.away_score, -1, 1)

    def datetime_converter(x):
        return datetime.strptime(x, "%Y-%m-%d")

    matches.date = matches.date.apply(datetime_converter)
    matches["month"] = matches.date.dt.month
    matches["day"] = matches.date.dt.day

    return matches


@st.cache_data
def load_schedule():
    schedule = pd.read_csv("data/schedule_2026.csv")
    schedule.columns = schedule.columns.str.strip().str.lower().str.replace(" ", "_")

    def datetime_converter(x):
        return datetime.strptime(x, "%Y-%m-%d")

    schedule.date = schedule.date.apply(datetime_converter)
    schedule.year -= 2026
    schedule["month"] = schedule.date.dt.month
    schedule["day"] = schedule.date.dt.day

    schedule["result"] = np.clip(schedule.home_score - schedule.away_score, -1, 1)

    return schedule


matches = load_matches()

st.sidebar.header("Select columns:")
columns = {
    column: True for column in ["home_team", "away_team", "year", "month", "day"]
}

for column in matches.columns.drop(["home_score", "away_score", "result"]):
    columns[column] = st.sidebar.checkbox(column, value=columns.get(column, False))

X = matches[[column for column in columns if columns[column]]]


prediction_type = st.selectbox(
    label="Select prediction:", options=["Scoreline", "Result"]
)


y_columns = (
    ["home_score", "away_score"] if prediction_type == "Scoreline" else ["result"]
)
y = matches[y_columns]

X_train, X_test, y_train, y_test = map(
    home_away_reverser, train_test_split(X, y, random_state=0)
)

X, y = map(home_away_reverser, [X, y])

categorical_features = [
    "home_team",
    "away_team",
    "round",
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

if prediction_type == "Scoreline":
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "Huber Regressor": MultiOutputRegressor(HuberRegressor(max_iter=1000)),
        "RANSAC Regressor": RANSACRegressor(random_state=0),
        "Random Forest Regressor": RandomForestRegressor(random_state=0),
        "Gradient Boosing Regressor": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=0)
        ),
        "KNN Regressor": KNeighborsRegressor(),
        "Neural Network": MLPRegressor(max_iter=100000, random_state=0),
    }

elif prediction_type == "Result":
    models = {
        "Random Forest Classifier": RandomForestClassifier(random_state=0),
        # "Logistic Regression": LogisticRegression(),
        # "Neural Network Classifier": MLPClassifier(random_state=0),
        # "1vRest Classifier": OneVsRestClassifier(LinearRegression()),
        # "Gradient Boosing Classifier": GradientBoostingClassifier(random_state=0),
        # "1v1 Classifier": OneVsOneClassifier(LinearRegression()),
    }

st.session_state.obs = {}

col1, col2 = st.columns(2)

with col1:
    st.session_state.obs["home_team"] = st.selectbox(
        label="Home", options=sorted(matches.home_team.unique())
    )

with col2:
    st.session_state.obs["away_team"] = st.selectbox(
        label="Away", options=sorted(matches.away_team.unique())
    )

if columns["round"]:
    st.session_state.obs["round"] = st.selectbox(
        label="Round", options=(matches["round"].unique())
    )

if columns["year"] or columns["month"] or columns["day"]:
    st.session_state.obs["selected_datetime"] = st.datetime_input("Date of match")
    st.session_state.obs["year"] = st.session_state.obs["selected_datetime"].year - 2026
    st.session_state.obs["month"] = st.session_state.obs["selected_datetime"].month
    st.session_state.obs["day"] = st.session_state.obs["selected_datetime"].day

cols = st.columns(len(models))
for i, col in enumerate(cols):
    with col:
        model_name, model = list(models.items())[i]
        pipeline = [("preprocess", preprocessor), ("model", model)]

        clf = Pipeline(pipeline)
        clf.fit(X, y)

        with st.container(height=2**7):
            st.write(model_name)

        schedule = load_schedule()
        pred = clf.predict(schedule)
        schedule_reversed = home_away_swap(schedule)
        pred_reversed = clf.predict(schedule_reversed)[:, ::-1]

        pred = np.round(np.mean([pred, pred_reversed], axis=0))

        # full_results_2026 = pd.concat([schedule, pd.DataFrame(pred, columns=["home_score_pred", "away_score_pred"])], axis=1)
        # full_results_2026

        pred_results = np.apply_along_axis(
            lambda x: 1 if x[0] > x[1] else 0 if x[0] == x[1] else -1, 1, pred
        ).reshape(-1, 1)
        pred = pd.DataFrame(
            np.concatenate([pred, pred_results], axis=1),
            columns=["home_score", "away_score", "result"],
        )
        correct = sum(
            (
                pred[["home_score", "away_score"]]
                == schedule[["home_score", "away_score"]]
            ).all(axis=1)
        )
        half_correct = sum(
            (
                pred[["home_score", "away_score"]]
                == schedule[["home_score", "away_score"]]
            ).any(axis=1)
        )
        correct_result = sum((pred["result"] == schedule["result"]))
        correct
        half_correct
        correct_result

        if prediction_type == "Scoreline":
            pred = clf.predict(pd.DataFrame(st.session_state.obs, index=[0]))[0]

            obs = st.session_state.obs.copy()
            obs["home_team"], obs["away_team"] = obs["away_team"], obs["home_team"]
            pred_reversed = clf.predict(pd.DataFrame(obs, index=[0]))[0][::-1]

            home_score = np.round(np.mean([pred[0], pred_reversed[0]])).astype(int)
            away_score = np.round(np.mean([pred[1], pred_reversed[1]])).astype(int)

            st.subheader("Predicted Score:")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label=f"{st.session_state.obs['home_team']}", value=home_score
                )

            with col2:
                st.metric(
                    label=f"{st.session_state.obs['away_team']}", value=away_score
                )

        elif prediction_type == "Result":
            st.subheader("Predicted Result:")
            pred = clf.predict(pd.DataFrame(st.session_state.obs, index=[0]))[0]

            obs = st.session_state.obs.copy()
            obs["home_team"], obs["away_team"] = obs["away_team"], obs["home_team"]
            pred_reversed = clf.predict(pd.DataFrame(obs, index=[0]))[0]

            pred = np.round(np.mean([pred, pred_reversed])).astype(int)

            if pred == 1:
                st.success("Home Wins")
            elif pred == 0:
                st.info("Draw")
            elif pred == -1:
                st.error("Away Wins")

# ---------- OLD ----------
# st.header("Old Dataset")
# # Data Cleaning
# matches = pd.read_csv("old_data/WorldCupMatches.csv")
# matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
# matches = matches.dropna()
# matches[
#     [
#         "year",
#         "home_team_goals",
#         "away_team_goals",
#         "half-time_home_goals",
#         "half-time_away_goals",
#         "roundid",
#         "matchid",
#     ]
# ] = matches[
#     [
#         "year",
#         "home_team_goals",
#         "away_team_goals",
#         "half-time_home_goals",
#         "half-time_away_goals",
#         "roundid",
#         "matchid",
#     ]
# ].astype(
#     int
# )
#
#
# def datetime_converter(x):
#     try:
#         return datetime.strptime(x, "%d %b %Y - %H:%M ")
#     except ValueError:
#         return datetime.strptime(x, "%d %B %Y - %H:%M ")
#
#
# matches.datetime = matches.datetime.apply(datetime_converter)
# matches["month"] = matches.datetime.dt.month
# matches["day"] = matches.datetime.dt.day
# matches = matches.drop("datetime", axis=1)
#
# matches.stage = matches.stage.replace(r"Group[\w\W]*", "Groups", regex=True)
# matches.home_team_initials = matches.home_team_initials.astype("category")
# matches.away_team_initials = matches.away_team_initials.astype("category")
# matches["result"] = np.clip(matches.home_team_goals - matches.away_team_goals, -1, 1)
#
# st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")
#
# # Column Selection
# st.sidebar.header("Select columns:")
# columns = {
#     column: True
#     for column in [
#         "year",
#         "half-time_home_goals",
#         "half-time_away_goals",
#         "home_team_initials",
#         "away_team_initials",
#         "month",
#         "day",
#     ]
# }
# for column in matches.columns:
#     columns[column] = st.sidebar.checkbox(column, value=columns.get(column, False))
#
# X = matches[[column for column in columns if columns[column]]]
#
#
#
# # Result Selection
# prediction_type = st.selectbox(
#     label="Select prediction:", options=["Scoreline", "Result"]
# )
#
# y_columns = (
#     ["home_team_goals", "away_team_goals"]
#     if prediction_type == "Scoreline"
#     else "result"
# )
# y = matches[y_columns]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# models = {
#     "Linear Regression": LinearRegression(),
#     "Neural Network Regressor": MLPRegressor(random_state=0),
#     "Random Forest Regressor": RandomForestRegressor(random_state=0),
#     "Bagging Regressor": BaggingRegressor(random_state=0),
#     "Extra Trees Regressor": ExtraTreesRegressor(random_state=0),
# }
# if prediction_type == "Result":
#     classifier_models = {
#         "Logistic Regression": LogisticRegression(),
#         "Neural Network Classifier": MLPClassifier(random_state=0),
#         "Random Forest Classifier": RandomForestClassifier(random_state=0),
#         "AdaBoost Classifier": AdaBoostClassifier(random_state=0),
#         "AdaBoost Regressor": AdaBoostRegressor(random_state=0),
#         "Bagging Classifier": BaggingClassifier(random_state=0),
#         "Extra Trees Classifier": ExtraTreesClassifier(random_state=0),
#         "Gradient Boosing Classifier": GradientBoostingClassifier(random_state=0),
#         "Gradient Boosing Regressor": GradientBoostingRegressor(random_state=0),
#         "Hist Gradient Boosting Classifier": HistGradientBoostingClassifier(
#             random_state=0
#         ),
#         "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(
#             random_state=0
#         ),
#     }
#
#     models |= classifier_models
#
# obs = {}
#
# col1, col2 = st.columns(2)
#
# with col1:
#     obs["home_team_initials"] = st.selectbox(
#         label="Home", options=sorted(matches.home_team_initials.unique())
#     )
#
#     if columns["half-time_home_goals"]:
#         obs["half-time_home_goals"] = st.number_input("Half Time Home Goals", step=1)
#
# with col2:
#     obs["away_team_initials"] = st.selectbox(
#         label="Away", options=sorted(matches.home_team_initials.unique())
#     )
#     if columns["half-time_away_goals"]:
#         obs["half-time_away_goals"] = st.number_input("Half Time Away Goals", step=1)
#
# now = datetime.now()
# if columns["year"] or columns["month"] or columns["year"]:
#     obs["selected_datetime"] = st.datetime_input("Date and time of match")
#     obs["year"] = obs["selected_datetime"].year
#     obs["month"] = obs["selected_datetime"].month
#     obs["day"] = obs["selected_datetime"].day
#
# cols = st.columns(len(models))
# for i, col in enumerate(cols):
#     with col:
#         model_name, model = list(models.items())[i]
#         pipeline = [("preprocess", preprocessor), ("model", model)]
#
#         clf = Pipeline(pipeline)
#         clf.fit(X_train, y_train)
#
#         st.subheader(model_name)
#         st.write(f"Accuracy: {clf.score(X_test, y_test)}")
#
#         clf = Pipeline(pipeline)
#         clf.fit(X, y)
#
#         pred = np.round(clf.predict(pd.DataFrame(obs, index=[0]))).astype(int)
#         if prediction_type == "Scoreline":
#             st.subheader("Predicted Score:")
#
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.metric(label=f"{obs['home_team_initials']}", value=pred[0][0])
#
#             with col2:
#                 st.metric(label=f"{obs['away_team_initials']}", value=pred[0][1])
#
#         elif prediction_type == "Result":
#             st.subheader("Predicted Result:")
#             pred = pred[0]
#             if pred == 1:
#                 st.success("Home Wins")
#             elif pred == 0:
#                 st.info("Draw")
#             elif pred == -1:
#                 st.error("Away Wins")
