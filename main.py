from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

st.set_page_config(page_title="FIFA World Cup 2026 Prediction Model", layout="wide")


@st.cache_data
def load_matches():
    matches = pd.read_csv("worldcup/data-csv/matches.csv").dropna()

    matches = matches[~matches.tournament_name.str.contains("Women")]

    def str_to_datetime(s):
        return datetime.strptime(s, "%Y-%m-%d")

    matches.match_date = matches.match_date.apply(str_to_datetime)
    matches["match_year"] = matches.match_date.dt.year - 2026
    matches["match_month"] = matches.match_date.dt.month
    matches["match_day"] = matches.match_date.dt.day
    matches["match_hour"] = matches.match_time.str.split(":").str[0].astype(int)
    matches["match_minute"] = matches.match_time.str.split(":").str[1].astype(int)

    st.header("Matches")
    st.dataframe(matches)

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
            "home_team_id",
            "home_team_name",
            "away_team_id",
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


data = matches

id_columns = data.columns[data.columns.str.endswith("id")]
data = data.drop(columns=id_columns)

data_reversed = data.rename(
    columns={
        "home_team_name": "away_team_name",
        "away_team_name": "home_team_name",
        "home_team_score": "away_team_score",
        "away_team_score": "home_team_score",
        "home_manager_name": "away_manager_name",
        "away_manager_name": "home_manager_name",
    }
)

data = pd.concat([data, data_reversed])

st.header("Data")
st.dataframe(data)

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

X = data[[column for column in columns if columns[column]]]
y = data[["home_team_score", "away_team_score"]]

st.header("X")
st.dataframe(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

mlb = MultiLabelBinarizer()

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
    MLPRegressor(max_iter=100000, random_state=0),
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

home_team_score, away_team_score = clf.predict(
    pd.DataFrame(obs, index=[0]).reindex(columns=clf.feature_names_in_)
)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(label=obs["home_team_name"], value=home_team_score)

with col2:
    st.metric(label=obs["away_team_name"], value=away_team_score)
