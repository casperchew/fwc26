import pandas as pd

old = pd.read_csv("data/schedule_2026_latest.csv")
new = pd.read_csv("worldcup/data-csv/matches.csv")

new = new.drop(new.index)

new.match_date = old.Date
new.match_time = old.Time
new.home_team_name = old.home_team
new.away_team_name = old.away_team
new.home_team_score = old.home_score
new.away_team_score = old.away_score

new.to_csv("new.csv", index=False)
