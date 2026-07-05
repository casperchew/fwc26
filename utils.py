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
