from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import home_away_symmetric, home_away_swap

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")


def load_matches(matches_2022=True, matches_2026=True) -> pd.DataFrame:
    matches = []

    if matches_2022:
        matches.append(pd.read_csv("worldcup/data-csv/matches.csv"))

    if matches_2026:
        matches.append(pd.read_csv("worldcup/data-csv/matches_2026.csv"))

    matches = pd.concat(matches).convert_dtypes()

    # matches = matches[~matches.tournament_name.str.contains("Women")]

    def str_to_datetime(s):
        return datetime.strptime(s, "%Y-%m-%d")

    matches.match_date = matches.match_date.apply(str_to_datetime)
    matches["match_year"] = matches.match_date.dt.year - 2026
    matches["match_month"] = matches.match_date.dt.month
    matches["match_day"] = matches.match_date.dt.day
    matches["match_hour"] = matches.match_time.str.split(":").str[0].astype(int)
    matches["match_minute"] = matches.match_time.str.split(":").str[1].astype(int)

    return matches[
        [
            "match_year",
            "match_month",
            "match_day",
            "match_hour",
            "match_minute",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
        ]
    ]

st.sidebar.header("Select dataset:")
matches_2022 = st.sidebar.checkbox(label="<2022")
matches_2026 = st.sidebar.checkbox(label="2026")

matches = load_matches(matches_2022=matches_2022, matches_2026=matches_2026)
data = matches

id_columns = data.columns[data.columns.str.endswith("id")]
data = data.drop(columns=id_columns)

st.sidebar.header("Select columns:")
columns = {
    column: True
    for column in [
        "match_year",
        "home_team_name",
        "away_team_name",
    ]
}

for column in data.columns:
    columns[column] = st.sidebar.checkbox(column, value=columns.get(column, False))


obs = {}

if (
    columns["match_year"]
    or columns["match_month"]
    or columns["match_day"]
    or columns["match_hour"]
    or columns["match_minute"]
):
    match_datetime = st.datetime_input(label="Datetime of match")
    obs["match_year"] = match_datetime.year - 2026
    obs["match_month"] = match_datetime.month
    obs["match_day"] = match_datetime.day
    obs["match_hour"] = match_datetime.hour
    obs["match_minute"] = match_datetime.minute

col1, col2 = st.columns(2)

with col1:
    obs["home_team_name"] = st.selectbox(
        label="Home",
        options=sorted(data.home_team_name.unique()),
    )

with col2:
    obs["away_team_name"] = st.selectbox(
        label="Away",
        options=sorted(data.away_team_name.unique()),
    )

X = data[[column for column in columns if columns[column]]]
y = data[["home_team_score", "away_team_score"]]

st.write(X)

clf = make_pipeline(
    # make_column_transformer(
    #     (
    #         FunctionTransformer(lambda x: np.sin(x / 12 * 2 * np.pi)),
    #         make_column_selector("match_month"),
    #     ),
    #     (
    #         FunctionTransformer(lambda x: np.sin(x / 31 * 2 * np.pi)),
    #         make_column_selector("match_day"),
    #     ),
    #     (
    #         FunctionTransformer(lambda x: np.sin(x / 24 * 2 * np.pi)),
    #         make_column_selector("match_hour"),
    #     ),
    #     (
    #         FunctionTransformer(lambda x: np.sin(x / 60 * 2 * np.pi)),
    #         make_column_selector("match_minute"),
    #     ),
    #     remainder="passthrough",
    # ),
    OneHotEncoder(handle_unknown="warn"),
    LinearRegression(),
    # MLPRegressor(max_iter=100000, random_state=0),
)

X = home_away_symmetric(X)
y = home_away_symmetric(y)

clf.fit(X, y)

pred = clf.predict(
    pd.DataFrame(obs, index=[0]).reindex(columns=clf.feature_names_in_)
)[0]

pred_reversed = clf.predict(
    home_away_swap(pd.DataFrame(obs, index=[0])).reindex(columns=clf.feature_names_in_)
)[0,::-1]

st.write(pred)

home_team_score, away_team_score = np.mean([pred, pred_reversed], axis=0)

col1, col2 = st.columns(2)

with col1:
    st.metric(label=obs["home_team_name"], value=home_team_score)

with col2:
    st.metric(label=obs["away_team_name"], value=away_team_score)
