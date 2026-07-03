import pandas as pd


def home_away_swap(data):
    data_reversed = data.rename(
        columns={
            "home_team": "away_team",
            "away_team": "home_team",
            "home_score": "away_score",
            "away_score": "home_score",
        }
    )

    if "result" in data.columns:
        data_reversed.result *= -1

    return data_reversed


def home_away_reverser(data):
    data_reversed = data.rename(
        columns={
            "home_team": "away_team",
            "away_team": "home_team",
            "home_score": "away_score",
            "away_score": "home_score",
        }
    )

    if "result" in data.columns:
        data_reversed.result *= -1

    return pd.concat([data, data_reversed])
