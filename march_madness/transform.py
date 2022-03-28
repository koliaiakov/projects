# Transform and unformize data

import pandas as pd
import numpy as np


def encode_teams_attributes(row, is_tournament):
    if row["LTeamID"] > row["WTeamID"]:
        (team1, team2, team1_win, team1_score, team2_score) = (
            row["WTeamID"],
            row["LTeamID"],
            1,
            row["WScore"],
            row["LScore"],
        )
    else:
        (team1, team2, team1_win, team1_score, team2_score) = (
            row["LTeamID"],
            row["WTeamID"],
            0,
            row["LScore"],
            row["WScore"],
        )

    if (team1_win == 1) & (row["WLoc"] == "H"):
        team1_home = 1
        team2_home = 0
    elif (team1_win == 0) & (row["WLoc"] == "A"):
        team2_home = 1
        team1_home = 0
    else:
        team1_home = 0
        team2_home = 0
    return (
        team1,
        team2,
        team1_win,
        team1_score,
        team2_score,
        team1_home,
        team2_home,
        is_tournament,
    )


def tuple_list_to_df(l, colnames=None):
    if l == []:
        return pd.DataFrame()
    else:
        if colnames is None:
            colnames = [str(i) for i in range(len(l[0]))]
    return pd.DataFrame(l, columns=colnames)


def complete_data(data):
    new_data = data.apply(
        lambda row: encode_teams_attributes(row, 1), axis=1
    ).values.tolist()
    new_data = tuple_list_to_df(
        new_data,
        colnames=[
            "team1",
            "team2",
            "team1_win",
            "team1_score",
            "team2_score",
            "team1_home",
            "team2_home",
            "is_tournament",
        ],
    )
    data_f = pd.concat([data, new_data], axis=1)
    return data_f


def format_game_row(row, teamid):
    if row["team1"] == teamid:
        res = row[
            [
                "Season",
                "is_tournament",
                "team1_home",
                "team1_score",
                "team2_score",
                "team1_win",
            ]
        ]
    else:
        res = row[
            [
                "Season",
                "is_tournament",
                "team2_home",
                "team2_score",
                "team1_score",
                "team1_win",
            ]
        ]
        res["team1_win"] = 1 - res["team1_win"]
    return res.to_list()


def get_last_5_games(
    teamid, season, daynum, team_games, kept_var, last_games=None, opponent=None
):
    if last_games is None:
        data = team_games[teamid]
        res = data[(data.Season <= season) & (data.DayNum < daynum)]
    else:
        res = last_games
    l = len(res)
    npf = []
    if len(res) < 6:
        return (None, None)
    else:
        for k in range(1, 6):
            res_k = res.iloc[l - k][kept_var]
            npf.append(format_game_row(res_k, teamid))
        if opponent is not None:
            res_opp = res[res.team2 == opponent]
            l = len(res_opp)
            if l == 0:
                return (np.array(npf), 0.5)
            else:
                n = min(l, 5)
                ratio = np.sum(res_opp.iloc[l : (l - n)]["team1_win"]) / n
                return (np.array(npf), ratio)
        else:
            return (np.array(npf), None)


def get_last_season_games(
    teamid,
    season,
    daynum,
    team_games,
    kept_var,
    opponent=None,
    default_score=71.88,
    last_games=None,
):
    if last_games is None:
        data = team_games[teamid]
        res = data[(data.Season == season) & (data.DayNum < daynum)]
    else:
        res = last_games[last_games.Season == season]
    l_res = len(res)
    npf = []
    if l_res == 0:
        return (
            np.array([None, None, None, default_score, default_score, 0.5]).reshape(
                1, 6
            ),
            0.5,
        )
    else:
        for k in range(1, l_res + 1):
            res_k = res.iloc[l_res - k][kept_var]
            npf.append(format_game_row(res_k, teamid))
        if opponent is not None:
            res_opp = res[res.team2 == opponent]
            l_opp = len(res_opp)
            if l_opp == 0:
                return (np.array(npf), 0.5)
            else:
                ratio = np.sum(res_opp["team1_win"]) / l_opp
                return (np.array(npf), ratio)
        else:
            return (np.array(npf), None)


def get_last_games(row, team_games, kept_var, index=None):
    print(index)
    tg1 = team_games[row["team1"]]
    t1_last_games = tg1[(tg1.Season <= row["Season"]) & (tg1.DayNum < row["DayNum"])]

    tg2 = team_games[row["team2"]]
    t2_last_games = tg2[(tg2.Season <= row["Season"]) & (tg2.DayNum < row["DayNum"])]

    t1, ratio = get_last_5_games(
        row["team1"],
        row["Season"],
        row["DayNum"],
        team_games,
        kept_var,
        opponent=row["team2"],
        last_games=t1_last_games,
    )
    t2, _ = get_last_5_games(
        row["team2"],
        row["Season"],
        row["DayNum"],
        team_games,
        kept_var,
        last_games=t2_last_games,
    )
    t1_season, ratio_season = get_last_season_games(
        row["team1"],
        row["Season"],
        row["DayNum"],
        team_games,
        kept_var,
        opponent=row["team2"],
        last_games=t1_last_games,
    )
    t2_season, _ = get_last_season_games(
        row["team2"],
        row["Season"],
        row["DayNum"],
        team_games,
        kept_var,
        last_games=t2_last_games,
    )
    l = []
    if (t1 is None) or (t2 is None):
        return None
    else:
        t1_victory_last5_games = np.mean(t1[:, 5])
        t2_victory_last5_games = np.mean(t2[:, 5])
        t1_score_last5_games = np.sum(t1[:, 3])
        t2_score_last5_games = np.sum(t2[:, 3])
        t1_oppscore_5_games = np.sum(t1[:, 4])
        t2_oppscore_5_games = np.sum(t2[:, 4])
        t1_victory_season_games = np.mean(t1_season[:, 5])
        t2_victory_season_games = np.mean(t2_season[:, 5])
        t1_score_season_games = np.mean(t1_season[:, 3])
        t2_score_season_games = np.mean(t2_season[:, 3])
        t1_oppscore_season_games = np.mean(t1_season[:, 4])
        t2_oppscore_season_games = np.mean(t2_season[:, 4])
        for k in range(5):
            l += list(t1[k, :])
            l += list(t2[k, :])
        l += [
            row["team1_home"],
            row["team2_home"],
            row["Season"],
            row["DayNum"],
            row["is_tournament"],
            ratio,
            t1_victory_last5_games,
            t2_victory_last5_games,
            t1_score_last5_games,
            t2_score_last5_games,
            t1_oppscore_5_games,
            t2_oppscore_5_games,
            t1_victory_season_games,
            t2_victory_season_games,
            t1_score_season_games,
            t2_score_season_games,
            t1_oppscore_season_games,
            t2_oppscore_season_games,
            ratio_season,
        ]
    return l


def build_global_data(data, is_tournament=False):
    nd = np.zeros(shape=(len(data) * 2, 11))
    tournament_dummy = 1 if is_tournament else 0
    for i, row in data.iterrows():
        nd[2 * i, 0] = row["Season"]
        nd[2 * i + 1, 0] = row["Season"]
        nd[2 * i, 1] = row["DayNum"]
        nd[2 * i + 1, 1] = row["DayNum"]
        nd[2 * i, 2] = row["WTeamID"]
        nd[2 * i, 3] = row["LTeamID"]
        nd[2 * i + 1, 2] = row["LTeamID"]
        nd[2 * i + 1, 3] = row["WTeamID"]
        nd[2 * i, 4] = 1
        nd[2 * i + 1, 4] = 0
        nd[2 * i, 5] = tournament_dummy
        nd[2 * i + 1, 5] = tournament_dummy
        nd[2 * i, 6] = 1 if row["WLoc"] == "H" else 0
        nd[2 * i, 7] = 1 if row["WLoc"] == "A" else 0
        nd[2 * i + 1, 6] = 1 if row["WLoc"] == "A" else 0
        nd[2 * i + 1, 7] = 1 if row["WLoc"] == "H" else 0
        nd[2 * i, 8] = row["WScore"]
        nd[2 * i + 1, 8] = row["LScore"]
        nd[2 * i, 9] = row["LScore"]
        nd[2 * i + 1, 9] = row["WScore"]
        nd[2 * i, 10] = row["WScore"] - row["LScore"]
        nd[2 * i + 1, 10] = row["LScore"] - row["WScore"]

    nd = pd.DataFrame(
        nd,
        columns=[
            "season",
            "daynym",
            "teamid",
            "opponentid",
            "win",
            "is_tournament",
            "is_home",
            "is_opp_home",
            "score",
            "opp_score",
            "diff_score",
        ],
    )

    return nd


def compute_stats(data):
    tmp = data.groupby(["season", "teamid"]).agg(
        nbwins=("win", "sum"),
        ratiowins=("win", "mean"),
        totalpoints=("score", "sum"),
        opptotalpoints=("opp_score", "sum"),
        oppmeanpoints=("opp_score", "mean"),
        avgdiff=("diff_score", "mean"),
    )
    results = {
        (row["season"], row["teamid"]): list(
            row[
                [
                    "nbwins",
                    "ratiowins",
                    "totalpoints",
                    "opptotalpoints",
                    "oppmeanpoints",
                    "avgdiff",
                ]
            ]
        )
        for _, row in tmp.reset_index().iterrows()
    }
    return results
