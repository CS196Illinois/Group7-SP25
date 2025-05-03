import pandas as pd
import requests
import os


API_KEY = '4b4fb050d6c044bd9a673435f98ed76b'
headers = {'X-Auth-Token': API_KEY}
url = 'https://api.football-data.org/v4/competitions/PL/matches?status=SCHEDULED'

response = requests.get(url, headers=headers)
data = response.json()
matches = data['matches']

import pandas as pd
import numpy as np
df = pd.read_csv("backend/20-21 EPL.csv")
df1 = pd.read_csv("backend/21-22 EPL.csv")
df2 = pd.read_csv("backend/22-23 EPL.csv")
df3 = pd.read_csv("backend/23-24 EPL.csv")
df4 = pd.read_csv("backend/24-25 EPL.csv")
frames = [df, df1, df2, df3, df4]
games = pd.concat(frames)
games.drop(columns=["Div", "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA", "PSH", "PSD", "PSA", "WHH", "WHD", "WHA", "VCH", "VCD", "VCA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA", "B365>2.5", "B365<2.5", "P>2.5", "P<2.5", "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5", "AHh", "B365AHH", "B365AHA", "PAHH", "PAHA", "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA", "B365CH", "B365CD", "B365CA", "BWCH", "BWCD", "BWCA", "IWCH", "IWCD", "IWCA", "PSCH", "PSCD", "PSCA", "WHCH", "WHCD", "WHCA", "VCCH", "VCCD", "VCCA", "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA", "B365C>2.5", "B365C<2.5", "PC>2.5", "PC<2.5", "MaxC>2.5", "MaxC<2.5", "AvgC>2.5", "AvgC<2.5", "AHCh", "B365CAHH", "B365CAHA", "PCAHH", "PCAHA", "MaxCAHH", "MaxCAHA", "AvgCAHH", "AvgCAHA", "BFH", "BFD", "BFA", "1XBH", "1XBD", "1XBA", "BFEH", "BFED", "BFEA", "BFE>2.5", "BFE<2.5", "BFEAHH", "BFEAHA", "BFCH", "BFCD", "BFCA", "1XBCH", "1XBCD", "1XBCA", "BFECH", "BFECD", "BFECA", "BFEC>2.5", "BFEC<2.5", "BFECAHH", "BFECAHA"], inplace=True)
games = games.reset_index()
games["Hour"] = games["Time"].str.replace(":.+", "", regex = True).astype("int")
games["Date"] = pd.to_datetime(games["Date"], dayfirst= True)
games["Home_field"] = 1


games["opp_code"] = games["AwayTeam"].astype("category").cat.codes
games["home_code"] = games["HomeTeam"].astype("category").cat.codes
games["result"] = games["FTR"].astype("category").cat.codes - 1

from sklearn.ensemble._forest import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = games[games["Date"] < "2024-08-16"]
test = games[games["Date"] > "2024-08-16"]
predictors = ["opp_code", "Hour", "home_code", "Home_field"]
rf.fit(train[predictors], train["result"])
preds = rf.predict(test[predictors])

team_name_map = {
    'Arsenal FC': 'Arsenal',
    'Aston Villa FC': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth',
    'Brentford FC': 'Brentford',
    'Brighton & Hove Albion FC': 'Brighton',
    'Burnley FC': 'Burnley',
    'Chelsea FC': 'Chelsea',
    'Crystal Palace FC': 'Crystal Palace',
    'Everton FC': 'Everton',
    'Fulham FC': 'Fulham',
    'Liverpool FC': 'Liverpool',
    'Luton Town FC': 'Luton',
    'Manchester City FC': 'Man City',
    'Manchester United FC': 'Man United',
    'Newcastle United FC': 'Newcastle',
    'Nottingham Forest FC': "Nott'm Forest",
    'Sheffield United FC': 'Sheffield Utd',
    'Tottenham Hotspur FC': 'Tottenham',
    'West Ham United FC': 'West Ham',
    'Wolverhampton Wanderers FC': 'Wolves',
    'West Bromwich Albion FC': 'West Brom',
    "Ipswich Town FC": "Ipswich",
    'Southampton FC': "Southampton",
    "Leicester City FC": 'Leicester'
}

# Define your prediction function using your model
def predict_match_result(data, home_team, away_team, match_hour):
    try:
        home_code = games[games["HomeTeam"] == home_team]["home_code"].iloc[0]
        opp_code = games[games["AwayTeam"] == away_team]["opp_code"].iloc[0]
    except IndexError:
        return "Error: team name not recognized."

    match_data = pd.DataFrame([{
        "opp_code": opp_code,
        "Hour": match_hour,
        "home_code": home_code,
        "Home_field": 1  
    }])

    match_data = match_data[["opp_code", "Hour", "home_code", "Home_field"]]
    prediction = rf.predict(match_data)[0]

    if prediction == -1:
        return f"{away_team} wins."
    elif prediction == 0:
        return "Draw."
    else:
        return f"{home_team} wins."

match_data = pd.DataFrame([{
    'homeTeam': team_name_map.get(m['homeTeam']['name'], m['homeTeam']['name']),
    'awayTeam': team_name_map.get(m['awayTeam']['name'], m['awayTeam']['name']),
    'utcDate': m['utcDate']
} for m in matches])

match_data['Date'] = pd.to_datetime(match_data['utcDate'])

match_data['Prediction'] = match_data.apply(
    lambda row: predict_match_result(games, row['homeTeam'], row['awayTeam'], row['Date'].hour),
    axis=1
)

games_html = ""
for _, row in match_data.iterrows():
    games_html += f"""
    <div class="game">
      <div class="teams">
        <img src="images/{row['homeTeam'].lower().replace(' ', '')}.png">
        <div class="vs">vs</div>
        <img src="images/{row['awayTeam'].lower().replace(' ', '')}.png">
      </div>
      <div class="prediction">
        Prediction: <span>{row['Prediction']}</span>
      </div>
      <a href="predictions.html" target="_blank" class="down-arrow">⬇</a>
    </div>
    """

# putting games on index.html
frontend_file_path = 'frontend/index.html'

if os.path.exists(frontend_file_path):
    with open(frontend_file_path, 'r', encoding='utf-8') as file:
        html_template = file.read()

    final_html = html_template.replace('{{games_here}}', games_html)

    with open(frontend_file_path, 'w', encoding='utf-8') as file:
        file.write(final_html)

    print("index.html updated with real predictions.")
else:
    print(f"Error: {frontend_file_path} not found.")
