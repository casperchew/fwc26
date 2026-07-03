from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
import streamlit as st
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")


@st.cache_data
def load_manager_appearances():
    manager_appearances = pd.read_csv(
        "worldcup/data-csv/manager_appearances.csv"
    ).dropna()

    return manager_appearances


@st.cache_data
def load_matches():
    matches = pd.read_csv("worldcup/data-csv/matches.csv").dropna()

    matches = matches[~matches.tournament_name.str.contains("Women")]

    def str_to_datetime(s):
        return datetime.strptime(s, "%Y-%m-%d")

    matches.match_date = matches.match_date.apply(str_to_datetime)
    matches["match_year"] = matches.match_date.dt.year
    matches["match_month"] = matches.match_date.dt.month
    matches["match_day"] = matches.match_date.dt.day
    matches["match_hour"] = matches.match_time.str.split(":").str[0].astype(int)
    matches["match_minute"] = matches.match_time.str.split(":").str[1].astype(int)

    return matches[
        [
            "tournament_name",
            "match_id",
            "stage_name",
            "match_year",
            "match_month",
            "match_day",
            "match_hour",
            "match_minute",
            "stadium_name",
            "city_name",
            "country_name",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
        ]
    ]


@st.cache_data
def load_player_appearances():
    player_appearances = pd.read_csv(
        "worldcup/data-csv/player_appearances.csv"
    ).dropna()

    return player_appearances


matches = load_matches()

# manager_appearances = load_manager_appearances()
# manager_appearances_grouped = manager_appearances.groupby(["match_id", "team_name"])[
#     "manager_id"
# ].apply(list)
# mlb = MultiLabelBinarizer()
# manager_appearances_binarized = mlb.fit_transform(manager_appearances_grouped)
# manager_appearances_binarized = pd.DataFrame(
#     manager_appearances_binarized,
#     columns=mlb.classes_,
#     index=manager_appearances_grouped.index,
# ).reset_index()
#
# st.dataframe(manager_appearances)
#
# manager_appearances_binarized["home_team_name"] = manager_appearances_binarized.team_name
# manager_appearances_binarized["away_team_name"] = manager_appearances_binarized.team_name
#
# manager_appearances = pd.concat([manager_appearances, manager_appearances_binarized])
#
# data = pd.merge(matches, manager_appearances, on=["match_id", "home_team_name"])
#
# st.dataframe(data)

st.sidebar.header("Select columns:")
columns = {
    column: True
    for column in [
        "stage_name",
        "match_year",
        "match_month",
        "match_day",
        "match_hour",
        "match_minute",
        "home_team_name",
        "away_team_name",
    ]
}

# for column in matches.columns.drop(
#     [
#         "score",
#         "home_team_score",
#         "away_team_score",
#         "home_team_score_margin",
#         "away_team_score_margin",
#         "extra_time",
#         "penalty_shootout",
#         "score_penalties",
#         "home_team_score_penalties",
#         "away_team_score_penalties",
#         "result",
#         "home_team_win",
#         "away_team_win",
#         "draw",
#     ]
# ):
for column in matches.columns:
    columns[column] = st.sidebar.checkbox(column, value=columns.get(column, False))

data = matches

X = data[[column for column in columns if columns[column]]]
y = data[["home_team_score", "away_team_score"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

mlb = MultiLabelBinarizer()

clf = make_pipeline(
    make_column_transformer(
        (
            FunctionTransformer(lambda x: np.sin(x / 12 * 2 * np.pi)),
            make_column_selector("match_month"),
        ),
        (
            FunctionTransformer(lambda x: np.sin(x / 31 * 2 * np.pi)),
            make_column_selector("match_day"),
        ),
        (
            FunctionTransformer(lambda x: np.sin(x / 24 * 2 * np.pi)),
            make_column_selector("match_hour"),
        ),
        (
            FunctionTransformer(lambda x: np.sin(x / 60 * 2 * np.pi)),
            make_column_selector("match_minute"),
        ),
        remainder="passthrough",
    ),
    OneHotEncoder(handle_unknown="warn"),
    LinearRegression(),
    # RandomForestRegressor(random_state=0),
    # MLPRegressor(max_iter=100000),
)

clf.fit(X_train, y_train)

obs = {}
if columns["stage_name"]:
    obs["stage_name"] = st.selectbox(
        label="",
        options=matches.stage_name.unique(),
        index=None,
        placeholder="Select Stage",
    )

if (
    columns["match_year"]
    or columns["match_month"]
    or columns["match_day"]
    or columns["match_hour"]
    or columns["match_minute"]
):
    match_datetime = st.datetime_input(label="")
    obs["match_year"] = match_datetime.year
    obs["match_month"] = match_datetime.month
    obs["match_day"] = match_datetime.day
    obs["match_hour"] = match_datetime.hour
    obs["match_minute"] = match_datetime.minute

col1, col2 = st.columns(2)

with col1:
    obs["home_team_name"] = st.selectbox(
        label="Home",
        options=sorted(matches.home_team_name.unique()),
    )

with col2:
    obs["away_team_name"] = st.selectbox(
        label="Away",
        options=sorted(matches.away_team_name.unique()),
    )

home_team_score, away_team_score = clf.predict(pd.DataFrame(obs, index=[0]))[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(label=obs["home_team_name"], value=home_team_score)

with col2:
    st.metric(label=obs["away_team_name"], value=away_team_score)
