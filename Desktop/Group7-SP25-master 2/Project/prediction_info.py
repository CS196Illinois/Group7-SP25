import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# copying code over from real.ipynb
df = pd.read_csv("backend/20-21 EPL.csv")
df1 = pd.read_csv("backend/21-22 EPL.csv")
df2 = pd.read_csv("backend/22-23 EPL.csv")
df3 = pd.read_csv("backend/23-24 EPL.csv")
df4 = pd.read_csv("backend/24-25 EPL.csv")
frames = [df, df1, df2, df3, df4]
games = pd.concat(frames)

games.drop(columns=[col for col in games.columns if any(x in col for x in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'Max', 'Avg', 'P', 'AH', '1XB', 'BF'])], inplace=True)
games = games.reset_index()
games["Hour"] = games["Time"].str.replace(":.+", "", regex=True).astype("int")
games["Date"] = pd.to_datetime(games["Date"], dayfirst=True)
games["Home_field"] = 1
games["opp_code"] = games["AwayTeam"].astype("category").cat.codes
games["home_code"] = games["HomeTeam"].astype("category").cat.codes
games["result"] = games["FTR"].astype("category").cat.codes - 1

features = ['Home_field', 'Hour', 'opp_code', 'home_code']

# Target variables
X = games[features]
y_home = games["FTHG"]
y_away = games["FTAG"]

# Train/test split
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

#  score prediction models
home_rf = RandomForestRegressor(n_estimators=100, random_state=42)
home_rf.fit(X_train, y_home_train)

away_rf = RandomForestRegressor(n_estimators=100, random_state=42)
away_rf.fit(X_train, y_away_train)

# Predict full time scores
games['predicted_home_score'] = home_rf.predict(X)
games['predicted_away_score'] = away_rf.predict(X)

# predict win percentages
games['home_win_pct'] = games['predicted_home_score'] / (games['predicted_home_score'] + games['predicted_away_score']) * 100
games['away_win_pct'] = games['predicted_away_score'] / (games['predicted_home_score'] + games['predicted_away_score']) * 100
games['draw_pct'] = 100 - (games['home_win_pct'] + games['away_win_pct'])

games['home_win_pct'] = games['home_win_pct'].round(2)
games['away_win_pct'] = games['away_win_pct'].round(2)
games['draw_pct'] = games['draw_pct'].round(2)

games['Prediction'] = games.apply(
    lambda row: 'Home Win' if row['home_win_pct'] > row['away_win_pct'] else 
                ('Away Win' if row['away_win_pct'] > row['home_win_pct'] else 'Draw'),
    axis=1
)

# predict home and away shots
if "HS" in games.columns and "AS" in games.columns:
    y_home_shots = games["HS"]
    y_away_shots = games["AS"]

    home_shots_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    away_shots_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    home_shots_rf.fit(X_train, y_home_shots.loc[X_train.index])
    away_shots_rf.fit(X_train, y_away_shots.loc[X_train.index])

    games["home_shots"] = home_shots_rf.predict(X).round().astype(int)
    games["away_shots"] = away_shots_rf.predict(X).round().astype(int)
else:
    print("HS/AS columns not found.")

# ✅ predict shots on target
if "HST" in games.columns and "AST" in games.columns:
    y_home_sot = games["HST"]
    y_away_sot = games["AST"]

    home_sot_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    away_sot_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    home_sot_rf.fit(X_train, y_home_sot.loc[X_train.index])
    away_sot_rf.fit(X_train, y_away_sot.loc[X_train.index])

    games["home_shots_on_target"] = home_sot_rf.predict(X).round().astype(int)
    games["away_shots_on_target"] = away_sot_rf.predict(X).round().astype(int)
else:
    print("HST/AST columns not found.")

# ✅ Predict Halftime Scores
if "HTHG" in games.columns and "HTAG" in games.columns:
    y_home_half = games["HTHG"]
    y_away_half = games["HTAG"]

    home_half_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    away_half_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    home_half_rf.fit(X_train, y_home_half.loc[X_train.index])
    away_half_rf.fit(X_train, y_away_half.loc[X_train.index])

    games["home_half_score"] = home_half_rf.predict(X).round().astype(int)
    games["away_half_score"] = away_half_rf.predict(X).round().astype(int)
else:
    print("HTHG/HTAG columns not found.")

# using predictions.html as a HTML template
with open("frontend/predictions.html", "r", encoding="utf-8") as file:
    html_template = file.read()

# create output folder
output_folder = "frontend/matches"
os.makedirs(output_folder, exist_ok=True)

# gnerate HTML files
for _, row in games.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    prediction = row["Prediction"]

    html = html_template.replace("{{home}}", home) \
        .replace("{{away}}", away) \
        .replace("{{prediction}}", prediction) \
        .replace("{{home_win_pct}}", str(row["home_win_pct"])) \
        .replace("{{draw_pct}}", str(row["draw_pct"])) \
        .replace("{{away_win_pct}}", str(row["away_win_pct"])) \
        .replace("{{home_score}}", str(row["predicted_home_score"])) \
        .replace("{{away_score}}", str(row["predicted_away_score"])) \
        .replace("{{home_half_score}}", str(row.get("home_half_score", 0))) \
        .replace("{{away_half_score}}", str(row.get("away_half_score", 0))) \
        .replace("{{home_shots}}", str(row.get("home_shots", 0))) \
        .replace("{{away_shots}}", str(row.get("away_shots", 0))) \
        .replace("{{home_shots_on_target}}", str(row.get("home_shots_on_target", 0))) \
        .replace("{{away_shots_on_target}}", str(row.get("away_shots_on_target", 0)))

    filename = f"{home.lower().replace(' ', '')}_vs_{away.lower().replace(' ', '')}.html"
    file_path = os.path.join(output_folder, filename)

    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write(html)

