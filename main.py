from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from pandas.core.common import random_state
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

    return matches[
        [
            "group_stage",
            "knockout_stage",
            "match_year",
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


@st.cache_resource
def get_trained_models(X, y):
    return [
        (
            "Linear Regression",
            make_pipeline(OneHotEncoder(handle_unknown="warn"), LinearRegression()).fit(
                X, y
            ),
        ),
        (
            "Neural Network",
            make_pipeline(
                OneHotEncoder(handle_unknown="warn"),
                MLPRegressor(max_iter=1000000, random_state=0),
            ).fit(X, y),
        ),
    ]


# Dataset options
st.sidebar.header("Select dataset:")
matches_2022 = st.sidebar.checkbox(label="<2022")
matches_2026 = st.sidebar.checkbox(label="2026")

data = load_matches(matches_2022=matches_2022, matches_2026=matches_2026)

id_columns = data.columns[data.columns.str.endswith("id")]
data = data.drop(columns=id_columns)

st.sidebar.header("Select columns:")
columns = {
    "match_year": True,
}

for column in columns:
    columns[column] = st.sidebar.checkbox(label=column, value=columns[column])


# Dataset
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

X = home_away_symmetric(X)
y = home_away_symmetric(y)


models = get_trained_models(X, y)


# Prediction options
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
    label="Group Stage or Knockout Stage",
    options=["Group Stage", "Knockout Stage"],
    index=1,
)
obs["group_stage"] = 1 * (group_or_knockout == "Group Stage")
obs["knockout_stage"] = 1 * (group_or_knockout == "Knockout Stage")

if columns["match_year"]:
    match_datetime = st.datetime_input(label="Datetime of match")
    obs["match_year"] = match_datetime.year - 2026


cols = st.columns(len(models))

for i, model in enumerate(models):
    with cols[i]:
        st.header(model[0])
        clf = model[1]

        # Prediction
        pred = pd.DataFrame(
            clf.predict(
                pd.DataFrame(obs, index=[0]).reindex(columns=clf.feature_names_in_)
            ),
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

        pred = pd.concat([pred, pred_reversed]).mean()
        pred[["home_team_win", "away_team_win", "draw"]] = softmax(
            pred[["home_team_win", "away_team_win", "draw"]]
        )
        pred = pred.round(2)

        pred_delta = pred
        if st.session_state.get(model[0]) is not None:
            pred_delta = (pred - st.session_state.get(model[0])).round(2)

        # Results
        st.subheader("Scoreline:")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=obs["home_team_name"],
                value=pred.home_team_score,
                delta=pred_delta.home_team_score,
            )

        with col2:
            st.metric(
                label=obs["away_team_name"],
                value=pred.away_team_score,
                delta=pred_delta.away_team_score,
            )

        st.metric(
            label="Chance of extra time:",
            value=pred.extra_time,
            delta=pred_delta.extra_time,
        )

        st.subheader("Penalty shootout:")
        st.metric(
            label="Chance of penalty shootout:",
            value=pred.penalty_shootout,
            delta=pred_delta.penalty_shootout,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=obs["home_team_name"],
                value=pred.home_team_score_penalties,
                delta=pred_delta.home_team_score_penalties,
            )

        with col2:
            st.metric(
                label=obs["away_team_name"],
                value=pred.away_team_score_penalties,
                delta=pred_delta.home_team_score_penalties,
            )

        st.subheader("Result probabilities:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=obs["home_team_name"],
                value=pred.home_team_win,
                delta=pred_delta.home_team_win,
            )

        with col2:
            st.metric(label="Draw", value=pred.draw, delta=pred_delta.draw)

        with col3:
            st.metric(
                label=obs["away_team_name"],
                value=pred.away_team_win,
                delta=pred_delta.away_team_win,
            )

        st.session_state[model[0]] = pred
