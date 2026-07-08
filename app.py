from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.core.common import random_state
from scipy.special import softmax
from sklearn import clone
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import home_away_swap, home_away_symmetric

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")


class Model:
    def __init__(self, name, scoreline_clf, penalty_clf):
        self.name = name
        self.scoreline_clf = scoreline_clf
        self.penalty_clf = penalty_clf


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
    X_penalty = X[y.penalty_shootout == 1]
    y_scoreline = y[
        [
            "home_team_score",
            "away_team_score",
            "extra_time",
            "penalty_shootout",
            "home_team_win",
            "away_team_win",
            "draw",
        ]
    ]
    y_penalty = y[y.penalty_shootout == 1][
        [
            "home_team_score_penalties",
            "away_team_score_penalties",
        ]
    ]

    trained_models = []
    for name, model in [
        ("Random Forest", RandomForestRegressor(random_state=0)),
        ("Neural Network", MLPRegressor(max_iter=1000000, random_state=0)),
    ]:
        trained_models.append(
            Model(
                name=name,
                scoreline_clf=make_pipeline(
                    OneHotEncoder(handle_unknown="warn"), clone(model)
                ).fit(X, y_scoreline),
                penalty_clf=make_pipeline(
                    OneHotEncoder(handle_unknown="warn"), clone(model)
                ).fit(X_penalty, y_penalty),
            )
        )

    return trained_models


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

scoreline_columns = [
    "home_team_score",
    "away_team_score",
    "extra_time",
    "penalty_shootout",
    "home_team_win",
    "away_team_win",
    "draw",
]
penalty_columns = [
    "home_team_score_penalties",
    "away_team_score_penalties",
]


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
        st.header(model.name)
        scoreline_clf = model.scoreline_clf
        penalty_clf = model.penalty_clf

        # Prediction
        pred_scoreline = pd.DataFrame(
            model.scoreline_clf.predict(
                pd.DataFrame(obs, index=[0]).reindex(
                    columns=scoreline_clf.feature_names_in_
                )
            ),
            columns=scoreline_columns,
        )

        pred_scoreline_reversed = home_away_swap(
            pd.DataFrame(
                scoreline_clf.predict(
                    home_away_swap(pd.DataFrame(obs, index=[0])).reindex(
                        columns=scoreline_clf.feature_names_in_
                    )
                ),
                columns=scoreline_columns,
            )
        )

        pred_penalty = pd.DataFrame(
            model.penalty_clf.predict(
                pd.DataFrame(obs, index=[0]).reindex(
                    columns=penalty_clf.feature_names_in_
                )
            ),
            columns=penalty_columns,
        )

        pred_penalty_reversed = home_away_swap(
            pd.DataFrame(
                model.penalty_clf.predict(
                    home_away_swap(pd.DataFrame(obs, index=[0])).reindex(
                        columns=penalty_clf.feature_names_in_
                    )
                ),
                columns=penalty_columns,
            )
        )

        pred_scoreline = pd.concat([pred_scoreline, pred_scoreline_reversed]).mean()
        pred_scoreline[["home_team_win", "away_team_win", "draw"]] = softmax(
            pred_scoreline[["home_team_win", "away_team_win", "draw"]]
        )

        pred_penalty = pd.concat([pred_penalty, pred_penalty_reversed]).mean()

        pred = pd.concat([pred_scoreline, pred_penalty]).round(2)
        pred[
            [
                "home_team_score",
                "away_team_score",
                "extra_time",
                "penalty_shootout",
                "home_team_score_penalties",
                "away_team_score_penalties",
            ]
        ] = pred[
            [
                "home_team_score",
                "away_team_score",
                "extra_time",
                "penalty_shootout",
                "home_team_score_penalties",
                "away_team_score_penalties",
            ]
        ].clip(
            0
        )

        pred_delta = pred
        if st.session_state.get(model.name) is not None:
            pred_delta = (pred - st.session_state.get(model.name)).round(2)

        st.session_state[model.name] = pred

        # Results
        fig, ax = plt.subplots(figsize=(16, 1))
        ax.axis("off")
        ax.barh(0, pred.home_team_win, color="#505b73")
        ax.barh(0, pred.draw, left=pred.home_team_win, color="#ff9f0a")
        ax.barh(
            0, pred.away_team_win, left=pred.home_team_win + pred.draw, color="#0088e7"
        )
        ax.text(
            pred.home_team_win / 2,
            0,
            f"{pred.home_team_win * 100:.0f}%",
            c="k",
            fontsize="xx-large",
            ha="center",
            va="center",
        )
        ax.text(
            pred.home_team_win + pred.draw / 2,
            0,
            f"{pred.draw * 100:.0f}%",
            c="k",
            fontsize="xx-large",
            ha="center",
            va="center",
        )
        ax.text(
            pred.home_team_win + pred.draw + pred.away_team_win / 2,
            0,
            f"{pred.away_team_win * 100:.0f}%",
            c="k",
            fontsize="xx-large",
            ha="center",
            va="center",
        )
        st.pyplot(fig, transparent=True)

        st.subheader("Predicted Scoreline:")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=obs["home_team_name"],
                value=pred.home_team_score.round(0).astype(int),
                delta=pred_delta.home_team_score,
            )

        with col2:
            st.metric(
                label=obs["away_team_name"],
                value=pred.away_team_score.round(0).astype(int),
                delta=pred_delta.away_team_score,
            )

        st.metric(
            label="Chance of extra time:",
            value=f"{pred.extra_time * 100:.0f}%",
            delta=pred_delta.extra_time,
        )

        st.subheader("Penalty shootout:")
        st.text(
            "Note: Penalty shootout predictions work better when including <2022 data due to the rarity of shootouts."
        )
        st.metric(
            label="Chance of penalty shootout:",
            value=f"{pred.penalty_shootout * 100:.0f}%",
            delta=pred_delta.penalty_shootout,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=obs["home_team_name"],
                value=pred.home_team_score_penalties.round(0).astype(int),
                delta=pred_delta.home_team_score_penalties,
            )

        with col2:
            st.metric(
                label=obs["away_team_name"],
                value=pred.away_team_score_penalties.round(0).astype(int),
                delta=pred_delta.home_team_score_penalties,
            )
