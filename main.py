from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import softmax
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import home_away_swap, home_away_symmetric

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")


def load_matches(matches_2022=True, matches_2026=True) -> pd.DataFrame:
    matches = []

    if matches_2022:
        matches_2022_df = pd.read_csv("worldcup/data-csv/matches.csv")
        matches_2022_df = matches_2022_df[
            ~matches_2022_df.tournament_name.str.contains("Women")
        ]
        matches.append(matches_2022_df)

    if matches_2026:
        matches.append(pd.read_csv("worldcup/data-csv/matches_2026.csv"))

    matches = pd.concat(matches).convert_dtypes()

    def str_to_datetime(s):
        return datetime.strptime(s, "%Y-%m-%d")

    matches.match_date = matches.match_date.apply(str_to_datetime)
    matches["match_year"] = matches.match_date.dt.year - 2026
    # matches["match_month"] = matches.match_date.dt.month
    # matches["match_day"] = matches.match_date.dt.day
    # matches["match_hour"] = matches.match_time.str.split(":").str[0].astype(int)
    # matches["match_minute"] = matches.match_time.str.split(":").str[1].astype(int)

    return matches[
        [
            "group_stage",
            "knockout_stage",
            "match_year",
            # "match_month",
            # "match_day",
            # "match_hour",
            # "match_minute",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
            "extra_time",
            "penalty_shootout",
            "home_team_score_penalties",
            "away_team_score_penalties",
            "home_team_win",
            "away_team_win",
            "draw",
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
    "match_year": True,
    # "match_month": False,
    # "match_day": False,
    # "match_hour": False,
    # "match_minute": False,
}

for column in columns:
    columns[column] = st.sidebar.checkbox(label=column, value=columns[column])


obs = {}

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

group_or_knockout = st.selectbox(
    label="Group Stage or Knockout Stage", options=["Group Stage", "Knockout Stage"]
)
obs["group_stage"] = 1 * (group_or_knockout == "Group Stage")
obs["knockout_stage"] = 1 * (group_or_knockout == "Knockout Stage")

if (
    columns["match_year"]
    # or columns["match_month"]
    # or columns["match_day"]
    # or columns["match_hour"]
    # or columns["match_minute"]
):
    match_datetime = st.datetime_input(label="Datetime of match")
    obs["match_year"] = match_datetime.year - 2026
    # obs["match_month"] = match_datetime.month
    # obs["match_day"] = match_datetime.day
    # obs["match_hour"] = match_datetime.hour
    # obs["match_minute"] = match_datetime.minute

X = data[
    [column for column in columns if columns[column]]
    + ["group_stage", "knockout_stage", "home_team_name", "away_team_name"]
]
y = data[
    [
        "home_team_score",
        "away_team_score",
        "extra_time",
        "penalty_shootout",
        "home_team_score_penalties",
        "away_team_score_penalties",
        "home_team_win",
        "away_team_win",
        "draw",
    ]
]

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

st.write(pd.DataFrame(obs, index=[0]).reindex(columns=clf.feature_names_in_))

pred = pd.DataFrame(
    clf.predict(pd.DataFrame(obs, index=[0]).reindex(columns=clf.feature_names_in_)),
    columns=y.columns,
)

pred_reversed = home_away_swap(
    pd.DataFrame(
        clf.predict(
            home_away_swap(pd.DataFrame(obs, index=[0])).reindex(
                columns=clf.feature_names_in_
            )
        ),
        columns=y.columns,
    )
)

pred = pd.concat([pred, pred_reversed]).mean().round(2)

st.write(pred)

st.subheader("Scoreline:")

col1, col2 = st.columns(2)

with col1:
    st.metric(label=obs["home_team_name"], value=pred.home_team_score)

with col2:
    st.metric(label=obs["away_team_name"], value=pred.away_team_score)

st.metric(label="Chance of extra time:", value=pred.extra_time)

st.subheader("Penalty shootout:")
st.metric(label="Chance of penalty shootout:", value=pred.penalty_shootout)

col1, col2 = st.columns(2)

with col1:
    st.metric(label=obs["home_team_name"], value=pred.home_team_score_penalties)

with col2:
    st.metric(label=obs["away_team_name"], value=pred.away_team_score_penalties)

st.subheader("Result probabilities:")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label=obs["home_team_name"], value=pred.home_team_win)

with col2:
    st.metric(label="Draw", value=pred.draw)

with col3:
    st.metric(label=obs["away_team_name"], value=pred.away_team_win)
