from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib  

app = Flask(__name__)
CORS(app)

team_rankings = {
    "Man City": 1, "Arsenal": 2, "Liverpool": 3, "Newcastle": 4, "Aston Villa": 5, "Tottenham": 6,
    "Man United": 7, "Brighton": 8, "Chelsea": 9, "Brentford": 10, "Fulham": 11, "Crystal Palace": 12,
    "Bournemouth": 13, "Nott'm Forest": 14, "West Ham": 15, "Everton": 16, "Wolves": 17, "Leeds": 18,
    "Leicester": 19, "Burnley": 20, "Luton": 21, "Ipswich": 22, "Norwich City": 23, "Southampton": 24, "Sheffield United": 25
}

# Load preprocessed data (make sure this part matches your training script!)
dataset = pd.read_csv("your_cleaned_dataset.csv")  # or just reuse previous preprocessing
predictors = dataset.select_dtypes(include=['int64']).drop(columns=["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
"HY",	"AY",	"HR",	"AR", "HTHG",	"HTAG", "index"])

target = dataset["FTR"]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(predictors)
model = LogisticRegression()
model.fit(train_scaled, target)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home = team_rankings.get(data["home"], 0)
    away = team_rankings.get(data["away"], 0)
    hour = int(data["hour"])

    input_data = {col: 0 for col in predictors.columns}
    input_data["HomeTeam"] = home
    input_data["AwayTeam"] = away
    input_data["Hour"] = hour

    df = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]

    if pred == 1:
        result = f"{data['home']} wins"
    elif pred == 0:
        result = "Draw"
    else:
        result = f"{data['away']} wins"

    return jsonify({"prediction": result})
