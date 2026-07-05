from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")

LATEST = True


@st.cache_data
def load_matches():
    matches = pd.read_csv("data/matches_1930_2022.csv")
    matches.columns = matches.columns.str.strip().str.lower().str.replace(" ", "_")
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

    def datetime_converter(x):
        return datetime.strptime(x, "%Y-%m-%d")

    matches.date = matches.date.apply(datetime_converter)
    matches["month"] = matches.date.dt.month
    matches["day"] = matches.date.dt.day

    return matches


@st.cache_data
def load_knockout():
    knockout = pd.read_csv("data/knockout_2026.csv")
    knockout.columns = knockout.columns.str.strip().str.lower().str.replace(" ", "_")
    knockout.year -= 2026

    return knockout


@st.cache_data
def load_schedule(latest=True):
    if latest:
        schedule = pd.read_csv("data/schedule_2026_latest.csv")
    else:
        schedule = pd.read_csv("data/schedule_2026.csv")

    schedule.columns = schedule.columns.str.strip().str.lower().str.replace(" ", "_")

    def datetime_converter(x):
        return datetime.strptime(x, "%Y-%m-%d")

    schedule.date = schedule.date.apply(datetime_converter)
    schedule.year -= 2026
    schedule["month"] = schedule.date.dt.month
    schedule["day"] = schedule.date.dt.day

    return schedule


matches = load_matches()
schedule = load_schedule(latest=LATEST)

# matches = pd.concat([matches, schedule])[
#     ["home_team", "away_team", "year", "home_score", "away_score"]
# ]
matches = schedule[["home_team", "away_team", "year", "home_score", "away_score"]]

matches = matches.replace("Bosnia-Herzegovina", "Bosnia and Herzegovina")
matches = matches.replace("Czech Republic", "Czechia")

st.sidebar.header("Select columns:")
columns = {column: True for column in ["home_team", "away_team", "year"]}

for column in matches.columns.drop(["home_score", "away_score"]):
    columns[column] = st.sidebar.checkbox(column, value=columns.get(column, False))

X = matches[[column for column in columns if columns[column]]]
y = matches[["home_score", "away_score"]]
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

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=0),
    "XGBoost": xgb.XGBRegressor(),
    "Neural Network": MLPRegressor(
        max_iter=1000000,
        random_state=0,
    ),
}


st.session_state.obs = {}

if LATEST:
    st.subheader("Using latest data")
else:
    st.subheader("Using groups data")

col1, col2 = st.columns(2)

with col1:
    st.session_state.obs["home_team"] = st.selectbox(
        label="Home", options=sorted(matches.home_team.unique())
    )

with col2:
    st.session_state.obs["away_team"] = st.selectbox(
        label="Away", options=sorted(matches.away_team.unique())
    )

if columns["year"]:
    selected_datetime = st.datetime_input("Date of match")
    st.session_state.obs["year"] = selected_datetime.year - 2026
    st.session_state.obs["month"] = selected_datetime.month
    st.session_state.obs["day"] = selected_datetime.day

cols = st.columns(len(models))
for i, col in enumerate(cols):
    with col:
        model_name, model = list(models.items())[i]
        pipeline = [("preprocess", preprocessor), ("model", model)]

        clf = Pipeline(pipeline)
        clf.fit(X, y)

        with st.container(height=2**7):
            st.write(model_name)

        pred = clf.predict(pd.DataFrame(st.session_state.obs, index=[0]))[0]

        obs = st.session_state.obs.copy()
        obs["home_team"], obs["away_team"] = obs["away_team"], obs["home_team"]
        pred_reversed = clf.predict(pd.DataFrame(obs, index=[0]))[0][::-1]

        home_score = np.mean([pred[0], pred_reversed[0]])
        away_score = np.mean([pred[1], pred_reversed[1]])

        st.header("Predicted Scoreline:")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=f"{st.session_state.obs['home_team']}",
                value=np.round(home_score).astype(int),
            )

        with col2:
            st.metric(
                label=f"{st.session_state.obs['away_team']}",
                value=np.round(away_score).astype(int),
            )

        st.header("Predicted Goals:")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=f"{st.session_state.obs['home_team']}",
                value=np.round(home_score, 2),
            )

        with col2:
            st.metric(
                label=f"{st.session_state.obs['away_team']}",
                value=np.round(away_score, 2),
            )

        st.header("Knockout predictions:")

        knockout = load_knockout()

        rounds = iter(["RO32", "RO16", "QF", "SF", "Finals"])
        while not knockout.empty:
            knockout_preds = clf.predict(knockout)
            knockout_preds_reversed = knockout.copy()
            (
                knockout_preds_reversed["home_team"],
                knockout_preds_reversed["away_team"],
            ) = (
                knockout_preds_reversed["away_team"],
                knockout_preds_reversed["home_team"],
            )
            knockout_preds_reversed = clf.predict(knockout_preds_reversed)[:, ::-1]

            knockout[["home_score", "away_score"]] = (
                knockout_preds + knockout_preds_reversed
            ) / 2

            try:
                r = next(rounds)
            except:
                break

            # if r in ["Finals"]:
            if True:
                st.subheader(r)
                for idx, row in knockout.iterrows():
                    if r == "Finals" and idx == 1:
                        st.subheader("3rd Place Match")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label=row.home_team, value=np.round(row.home_score, 2)
                        )

                    with col2:
                        st.metric(
                            label=row.away_team, value=np.round(row.away_score, 2)
                        )

            if knockout.shape[0] == 1:
                break

            knockout.loc[knockout.home_score >= knockout.away_score, "winner"] = (
                knockout.home_team
            )
            knockout.loc[knockout.home_score < knockout.away_score, "winner"] = (
                knockout.away_team
            )

            winners = list(knockout.winner)

            if r == "SF":
                knockout.loc[knockout.home_score >= knockout.away_score, "loser"] = (
                    knockout.away_team
                )
                knockout.loc[knockout.home_score < knockout.away_score, "loser"] = (
                    knockout.home_team
                )
                winners += list(knockout.loser)

            knockout = pd.DataFrame(
                {
                    "home_team": winners[0::2],
                    "away_team": winners[1::2],
                },
                columns=knockout.columns,
            )
            knockout.year = 0


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
