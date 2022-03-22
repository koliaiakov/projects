import pandas as pd
import numpy as np
import sys

sys.path.insert(1, "march_madness/")
from transform import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import argparse

parser = argparse.ArgumentParser(prog="main")
parser.add_argument(
    "--test", action="store_true", help="Quick test process with less data"
)
parser.add_argument(
    "--model",
    action="store",
    help="Indicate the type of model you want to train and test (keras_nn, gboosting, rf)",
)
args = vars(parser.parse_args())  # creation of a dictionnary

tournament_results = pd.read_csv("tournament_results.csv")
rs_results = pd.read_csv("regular_season_results.csv")
teams = pd.read_csv("teams.csv")

# tournament_results = pd.read_csv("data/WNCAATourneyCompactResults.csv")
# rs_results = pd.read_csv("data/WRegularSeasonCompactResults.csv")
# teams = pd.read_csv("data/WTeams.csv")

tournament_results = complete_data(tournament_results)
rs_results = complete_data(rs_results)

data = pd.concat([tournament_results, rs_results], axis=0)
data_sorted = data.sort_values(by=["Season", "DayNum"]).reset_index(drop=True)
team_games = {
    u: data_sorted[(data_sorted.team1 == u) | (data_sorted.team2 == u)]
    for u in teams["TeamID"]
}

if args["test"]:
    print("Test process with 5000 rows only")
    data_sorted = data_sorted.iloc[1:5000]

df = pd.DataFrame()
df["labels"] = data_sorted["team1_win"]
var = [
    "Season",
    "DayNum",
    "NumOT",
    "is_tournament",
    "team1",
    "team2",
    "team1_home",
    "team2_home",
    "team1_score",
    "team2_score",
    "team1_win",
]
df["values"] = [
    get_last_games(row, team_games, var, i) for i, row in data_sorted.iterrows()
]
df1 = df.dropna().sample(frac=1).reset_index(drop=True)

X_np = np.zeros(shape=(len(df1), 79))
for k in range(len(df1["values"])):
    X_np[k, :] = df1["values"].iloc[k]
X_np = normalize(X_np)
Y_np = np.array(df1["labels"]).astype(np.float).reshape(len(df1), 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_np, Y_np, test_size=0.2, random_state=0
)

if args["model"] == "keras_nn":
    print("Testing Keras NN model")
    model = keras_mlp(
        [(30, "relu"), (30, "relu"), (1, "sigmoid")],
        "binary_crossentropy",
        input_dim=79,
    )

    model = keras_mlp_fit(model, X_train, Y_train, 50)
    keras_mlp_evaluate(model, X_test, Y_test)

elif args["model"] == "gboosting":
    print("Testing Sklearn GradientBoosting model")
    model = sklearn_boosting(100)
    model = sklearn_fit(model, X_train, Y_train)
    sklearn_evaluate(model, X_test, Y_test)
elif args["model"] == "rf":
    print("Testing Sklearn RandomForest model")
    model = sklearn_rf(100)
    model = sklearn_fit(model, X_train, Y_train)
    sklearn_evaluate(model, X_test, Y_test)
else:
    raise ("ERROR wrong model choice " + args["model"])
