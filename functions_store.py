from safe_requests import safe_request
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
import shap
import datetime
from scipy.optimize import linear_sum_assignment
from sklearn.utils.class_weight import compute_sample_weight
import math
import itertools

key = {
    "RB": 0,
    "LB": 1,
    "CB": 2,
    "DM": 3,
    "RM": 4,
    "LM": 5,
    "CM": 6,
    "RW": 7,
    "LW": 8,
    "AM": 9,
    "ST": 10
}

statCategories = ['accurateCross', 'accurateLongBalls', 'accurateOppositionHalfPasses', 'accurateOwnHalfPasses',
                  'accuratePass', 'aerialLost', 'aerialWon', 'averageX', 'averageY', 'ballRecovery', 'bigChanceCreated',
                  'bigChanceMissed', 'blockedScoringAttempt', 'challengeLost', 'clearanceOffLine', 'dispossessed',
                  'duelLost', 'duelWon', 'errorLeadToAGoal', 'errorLeadToAShot', 'expectedAssists', 'expectedGoals',
                  'expectedGoalsOnTarget', 'fouls', 'goalAssist', 'goals', 'hitWoodwork', 'interceptionWon', 'keyPass',
                  'lastManTackle', 'minutesPlayed', 'onTargetScoringAttempt', 'outfielderBlock', 'ownGoals',
                  'penaltyConceded', 'penaltyMiss', 'penaltyWon', 'position', 'possessionLostCtrl', 'rating',
                  'ratingVersions', 'shotOffTarget', 'totalClearance', 'totalContest', 'totalCross', 'totalLongBalls',
                  'totalOffside', 'totalOppositionHalfPasses', 'totalOwnHalfPasses', 'totalPass', 'totalShots',
                  'totalTackle', 'touches', 'unsuccessfulTouch', 'wasFouled', 'wonContest', 'wonTackle']
statCategories.remove('goals')

lineups = {}

def getFormation(homeaway):
    formation = []
    if lineups[homeaway]['formation'] == '3-1-4-2':
        formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '3-2-4-1':
        formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST']
    elif lineups[homeaway]['formation'] == '3-3-1-3':
        formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'LM', 'AM', 'RW', 'ST', 'LW']
    elif lineups[homeaway]['formation'] == '3-3-3-1':
        formation = ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'RW', 'AM', 'LW', 'ST']
    elif lineups[homeaway]['formation'] == '3-4-1-2':
        formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '3-4-2-1':
        formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'AM', 'ST']
    elif lineups[homeaway]['formation'] == '3-4-3':
        formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'RW', 'ST', 'LW']
    elif lineups[homeaway]['formation'] == '3-5-1-1':
        formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'AM', 'ST']
    elif lineups[homeaway]['formation'] == '3-5-2':
        formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '4-1-3-2':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'CM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '4-1-4-1':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST']
    elif lineups[homeaway]['formation'] == '4-2-2-2':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '4-2-3-1':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'AM', 'LW', 'ST']
    elif lineups[homeaway]['formation'] == '4-2-4':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'RW', 'ST', 'ST', 'LW']
    elif lineups[homeaway]['formation'] == '4-3-1-2':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '4-3-2-1':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST']
    elif lineups[homeaway]['formation'] == '4-3-3':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'RW', 'ST', 'LW']
    elif lineups[homeaway]['formation'] == '4-4-1-1':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST']
    elif lineups[homeaway]['formation'] == '4-4-2':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '4-5-1':
        formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST']
    elif lineups[homeaway]['formation'] == '5-3-2':
        formation = ['GK', 'RB', 'CB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'ST', 'ST']
    elif lineups[homeaway]['formation'] == '5-4-1':
        formation = ['GK', 'RB', 'CB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST']
    return formation

def jsonRequest(url, cache_ttl=None, defaultSession=None):
    if cache_ttl is None:
        resp = safe_request(url, defaultSession=defaultSession)  # uses default CACHE_TTL
    else:
        cache_ttl = cache_ttl * 3600
        resp = safe_request(url, cache_ttl=cache_ttl, defaultSession=defaultSession)
    return resp.json()


def importAndTransformDict(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    for x in data.copy().keys():
        if 'minutesPlayed' not in data[x] or data[x]['position'] == 'GK':
            data.pop(x)

    for n in data.keys():
        for m in statCategories:
            if m not in data[n]:
                data[n][m] = 0

    df = pd.DataFrame(data).T
    return df


def MLModel():
    combinedDF = pd.concat(
        (importAndTransformDict("data/positionalStatsPrem2223.json"),
         importAndTransformDict("data/positionalStatsPrem2324.json"),
         importAndTransformDict("data/positionalStatsPrem2425.json")), ignore_index=True)
    combinedDF = combinedDF[statCategories]
    X = combinedDF.drop(columns=['position', 'ratingVersions'])
    X[list(X.columns)] = combinedDF[list(X.columns)].apply(pd.to_numeric, errors='coerce')

    y_position = combinedDF['position']

    y = y_position.map(key)
    y = y.apply(pd.to_numeric, errors='coerce')

    sampleWeights = compute_sample_weight(class_weight='balanced', y=y)

    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        max_depth=4,
        learning_rate=0.1,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y, sample_weight=sampleWeights)

    return model, X


def matchProbs(matchId, model, X):
    lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups")

    homeLineups = lineups['home']['players']
    awayLineups = lineups['away']['players']

    homePlayerStats = []

    count = 0
    for player in homeLineups:
        if player['position'] != 'G' and count < 10:
            count += 1
            homePlayerStats.append(player['statistics'])

    awayPlayerStats = []

    count = 0
    for player in awayLineups:
        if player['position'] != 'G' and count < 10:
            count += 1
            awayPlayerStats.append(player['statistics'])

    for n in range(len(homePlayerStats)):
        for m in statCategories:
            if m not in homePlayerStats[n]:
                homePlayerStats[n][m] = 0

    for n in range(len(awayPlayerStats)):
        for m in statCategories:
            if m not in awayPlayerStats[n]:
                awayPlayerStats[n][m] = 0

    columns = list(X.columns)

    homePlayerStats = pd.DataFrame(homePlayerStats)
    homePlayerStats = homePlayerStats[columns]
    homePlayerStats = homePlayerStats.apply(pd.to_numeric, errors='coerce')

    awayPlayerStats = pd.DataFrame(awayPlayerStats)
    awayPlayerStats = awayPlayerStats[columns]
    awayPlayerStats = awayPlayerStats.apply(pd.to_numeric, errors='coerce')

    homeProb = model.predict_proba(homePlayerStats)
    awayProb = model.predict_proba(awayPlayerStats)
    return homeProb, awayProb


def formation_likelihood(prob_matrix, formation, positions, return_mapping=False):
    """
    Improved two-stage formation likelihood:
    1. Locks in players whose top predicted positions match positions in the formation,
       prioritizing higher probabilities (not list order).
    2. Uses Hungarian algorithm to optimally assign remaining players to remaining positions.
    3. Returns total absolute probability and optional mapping.
    """

    n_players = prob_matrix.shape[0]
    slots = list(formation)
    position_indices = [positions.index(pos) for pos in slots]

    # --- Stage 1: Identify overlaps (but sort them by confidence first) ---
    player_top_pos = np.argmax(prob_matrix, axis=1)
    top_probs = np.max(prob_matrix, axis=1)

    # Sort players by how confident the model is about their top position
    player_order = sorted(range(n_players), key=lambda i: top_probs[i], reverse=True)

    locked_players = []
    locked_positions = []
    remaining_players = list(range(n_players))
    remaining_slots = list(range(len(slots)))

    for player_idx in player_order:
        top_pos = positions[player_top_pos[player_idx]]
        if top_pos in slots:
            # Find an available formation slot for that position
            for slot_idx in remaining_slots:
                if slots[slot_idx] == top_pos:
                    locked_players.append(player_idx)
                    locked_positions.append(slot_idx)
                    remaining_players.remove(player_idx)
                    remaining_slots.remove(slot_idx)
                    break  # move to next player once locked

    # --- Stage 2: Optimal assignment for remaining players/positions ---
    assigned_pairs = []
    if remaining_players and remaining_slots:
        submatrix = prob_matrix[np.ix_(remaining_players, [position_indices[i] for i in remaining_slots])]
        cost_matrix = 1 - submatrix  # higher probability = lower cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_pairs = list(zip(
            [remaining_players[r] for r in row_ind],
            [remaining_slots[c] for c in col_ind]
        ))

    # Combine overlaps and assignments
    all_pairs = list(zip(locked_players, locked_positions)) + assigned_pairs

    # --- Stage 3: Compute total probability ---
    total_prob = 1.0
    for player_idx, slot_idx in all_pairs:
        pos_index = positions.index(slots[slot_idx])
        prob = prob_matrix[player_idx, pos_index]
        total_prob *= prob

    # --- Stage 4: Optional mapping output ---
    if return_mapping:
        mapping = [
            (f"Player {p + 1}", slots[s], float(prob_matrix[p, positions.index(slots[s])]))
            for p, s in all_pairs
        ]
        return total_prob, mapping

    return total_prob


def round_sig(x, sig=3):
    """
    Round to a given number of significant figures and
    display in fixed-point decimal notation (no scientific notation).
    """
    if x == 0:
        return "0"
    # round numerically to significant figures
    rounded = round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))
    # format as fixed-point (avoiding scientific notation)
    formatted = f"{rounded:.{sig + 6}f}".rstrip("0").rstrip(".")
    # ensure there's always a 0 before the decimal if number < 1
    if formatted.startswith("."):
        formatted = "0" + formatted
    return formatted


def scale_ratings_to_10(ratings, give=0):
    """
    Scales a list or array of player ratings to the range [0, 10].
    """
    ratings = np.array(ratings, dtype=float)
    min_r, max_r = np.min(ratings), np.max(ratings)

    if min_r == max_r:
        return np.full_like(ratings, 5)

    scaled = give + (10 - give) * (ratings - min_r) / (max_r - min_r)
    return scaled


def importJson(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data


def getFullTeamName(name):
    data = jsonRequest(f"http://www.sofascore.com/api/v1/search/teams?q={name}&page=0")['results']
    count = 0
    for i in data:
        if count < 10 and 'gender' in i['entity']:
            count += 1
            print(f"{count}. {i['entity']['name']} ({i['entity']['gender']})")
    choice = input("Which number would you like to choose? ")
    return data[int(choice) - 1]['entity']['name']


def zeroDivide(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator
