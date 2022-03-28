import pandas as pd
import numpy as np
import sys

sys.path.insert(1, "march_madness/")
from transform import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class data:
    # mixed model means we put tournament and regular season together in a whole dataset
    # in the other case we use regular season data to build features for the model trained
    # on tournament games only
    def __init__(self, mixed_model=True, test=False):
        self.test = test
        # self.tournament_results = pd.read_csv("data/WNCAATourneyCompactResults.csv")
        self.tournament_results = pd.read_csv("data/tournament_results.csv")
        # self.rs_results = pd.read_csv("data/WRegularSeasonCompactResults.csv")
        self.rs_results = pd.read_csv("data/regular_season_results.csv")
        # self.teams = pd.read_csv("data/WTeams.csv")
        self.teams = pd.read_csv("data/teams.csv")
        print("Process mixed model", mixed_model)
        self.mixed_model = mixed_model
        if not mixed_model:
            self.seeds = pd.read_csv("data/seeds.csv")
            #self.seeds = pd.read_csv("data/WNCAATourneySeeds.csv")

    def build_mixed_model_dataset(self):
        tournament_results = complete_data(self.tournament_results)
        rs_results = complete_data(self.rs_results)
        data = pd.concat([tournament_results, rs_results], axis=0)
        data_sorted = data.sort_values(by=["Season", "DayNum"]).reset_index(drop=True)
        team_games = {
            u: data_sorted[(data_sorted.team1 == u) | (data_sorted.team2 == u)]
            for u in self.teams["TeamID"]
        }

        if self.test:
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

        X = tuple_list_to_df(list(df1["values"]))
        df_final = pd.concat([pd.DataFrame(df1["labels"]), X], axis=1)
        return df_final

    def build_diff_model_dataset(self):
        t_data = build_global_data(self.tournament_results, is_tournament=True)
        rs_data = build_global_data(self.rs_results, is_tournament=False)
        seeds = {}
        for _, row in self.seeds.iterrows():
            seeds[(row["Season"], row["TeamID"])] = row["Seed"][1:]
        stats = compute_stats(rs_data)

        df_final = pd.DataFrame()
        for _, row in t_data.iterrows():
            res1 = stats[(row["season"], row["teamid"])]
            res2 = stats[(row["season"], row["opponentid"])]
            seed1 = seeds[(row["season"], row["teamid"])]
            seed2 = seeds[(row["season"], row["opponentid"])]
            res = [row["win"]] + [seed1] + res1 + [seed2] + res2
            res = pd.DataFrame(np.array(res).reshape(1, 15))
            df_final = pd.concat([df_final, pd.DataFrame(res)], axis=0)

        return df_final

    def build_data_for_model(self, data, test_size=0.2):
        x = data.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        df = df.dropna().sample(frac=1).reset_index(drop=True)
        X_train, X_test, Y_train, Y_test = train_test_split(
            df.iloc[:, 1:], df[0], test_size=test_size, random_state=0
        )
        return (X_train, X_test, Y_train, Y_test)
