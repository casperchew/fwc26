import pandas as pd


def home_away_rename(s):
    if "home" in s:
        return s.replace("home", "away")

    elif "away" in s:
        return s.replace("away", "home")

    else:
        return s


def home_away_swap(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=home_away_rename)


def home_away_symmetric(df: pd.DataFrame):
    return pd.concat([df, df.rename(columns=home_away_rename)])


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
