import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from safe_requests import safe_request
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
from functions_store import round_sig, getFullTeamName, MLModel

leagueId = 17
seasonId = 61627


# seasonId = 52186
# seasonId = 41886

def jsonRequest(url, cache_ttl=None):
    if cache_ttl is None:
        resp = safe_request(url)  # uses default CACHE_TTL
    else:
        cache_ttl = cache_ttl * 3600
        resp = safe_request(url, cache_ttl=cache_ttl)
    return resp.json()


def getSeasonStatList(leagueId, seasonId):
    gamesPlayed = []
    matchIds = []
    stats = {}
    standings = \
        jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/standings/total")[
            'standings'][0]['rows']
    for x in standings:
        gamesPlayed.append(x['matches'])
    rounds = max(gamesPlayed)
    for round in range(rounds):
        roundEvents = jsonRequest(
            f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/{round + 1}",
            cache_ttl=24 * 14)[
            'events']
        for event in roundEvents:
            if safe_request(f"http://www.sofascore.com/api/v1/event/{event['id']}/lineups",
                            cache_ttl=0).status_code == 200 and safe_request(
                f"http://www.sofascore.com/api/v1/event/{event['id']}/average-positions",
                cache_ttl=0).status_code == 200:
                matchIds.append(event['id'])
    for matchId in matchIds:
        lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", cache_ttl=0)
        averagePositions = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/average-positions",
                                       cache_ttl=0)
        for x in lineups['home']['players']:
            x = lineups['home']['players'].index(x)
            if x < 11:
                print(x)
                print(lineups['home']['formation'])
                player = lineups['home']['players'][x]
                for y in averagePositions['home']:
                    if player['player']['id'] == y['player']['id']:
                        averageX = y['averageX']
                        averageY = y['averageY']
                formation = []
                if lineups['home']['formation'] == '3-1-4-2':
                    formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
                elif lineups['home']['formation'] == '3-2-4-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST']
                elif lineups['home']['formation'] == '3-3-1-3':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'LM', 'AM', 'RW', 'ST', 'LW']
                elif lineups['home']['formation'] == '3-3-3-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'RW', 'AM', 'LW', 'ST']
                elif lineups['home']['formation'] == '3-4-1-2':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST', 'ST']
                elif lineups['home']['formation'] == '3-4-2-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'AM', 'ST']
                elif lineups['home']['formation'] == '3-4-3':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'RW', 'ST', 'LW']
                elif lineups['home']['formation'] == '3-5-1-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'AM', 'ST']
                elif lineups['home']['formation'] == '3-5-2':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST', 'ST']
                elif lineups['home']['formation'] == '4-1-4-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST']
                elif lineups['home']['formation'] == '4-2-2-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST']
                elif lineups['home']['formation'] == '4-2-3-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'AM', 'LW', 'ST']
                elif lineups['home']['formation'] == '4-2-4':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'RW', 'ST', 'ST', 'LW']
                elif lineups['home']['formation'] == '4-3-1-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST']
                elif lineups['home']['formation'] == '4-3-2-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST']
                elif lineups['home']['formation'] == '4-3-3':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'RW', 'ST', 'LW']
                elif lineups['home']['formation'] == '4-4-1-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST']
                elif lineups['home']['formation'] == '4-4-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
                elif lineups['home']['formation'] == '4-5-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST']
                elif lineups['home']['formation'] == '5-3-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'ST', 'ST']
                elif lineups['home']['formation'] == '5-4-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST']
                playerMatchId = int(str(player['player']['id']) + str(matchId))
                player['statistics']['averageX'] = averageX
                player['statistics']['averageY'] = averageY
                player['statistics']['position'] = formation[x]
                stats[playerMatchId] = player['statistics']
        for x in lineups['away']['players']:
            x = lineups['away']['players'].index(x)
            if x < 11:
                print(x)
                print(lineups['away']['formation'])
                player = lineups['away']['players'][x]
                for y in averagePositions['away']:
                    if player['player']['id'] == y['player']['id']:
                        averageX = y['averageX']
                        averageY = y['averageY']
                formation = []
                if lineups['away']['formation'] == '3-1-4-2':
                    formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
                elif lineups['away']['formation'] == '3-2-4-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST']
                elif lineups['away']['formation'] == '3-3-1-3':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'LM', 'AM', 'RW', 'ST', 'LW']
                elif lineups['away']['formation'] == '3-3-3-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'RW', 'AM', 'LW', 'ST']
                elif lineups['away']['formation'] == '3-4-1-2':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST', 'ST']
                elif lineups['away']['formation'] == '3-4-2-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'AM', 'ST']
                elif lineups['away']['formation'] == '3-4-3':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'RW', 'ST', 'LW']
                elif lineups['away']['formation'] == '3-5-1-1':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'AM', 'ST']
                elif lineups['away']['formation'] == '3-5-2':
                    formation = ['GK', 'CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST', 'ST']
                elif lineups['away']['formation'] == '4-1-4-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST']
                elif lineups['away']['formation'] == '4-2-2-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST']
                elif lineups['away']['formation'] == '4-2-3-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'AM', 'LW', 'ST']
                elif lineups['away']['formation'] == '4-2-4':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'RW', 'ST', 'ST', 'LW']
                elif lineups['away']['formation'] == '4-3-1-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST']
                elif lineups['away']['formation'] == '4-3-2-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST']
                elif lineups['away']['formation'] == '4-3-3':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'RW', 'ST', 'LW']
                elif lineups['away']['formation'] == '4-4-1-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST']
                elif lineups['away']['formation'] == '4-4-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST']
                elif lineups['away']['formation'] == '4-5-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST']
                elif lineups['away']['formation'] == '5-3-2':
                    formation = ['GK', 'RB', 'CB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'ST', 'ST']
                elif lineups['away']['formation'] == '5-4-1':
                    formation = ['GK', 'RB', 'CB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST']
                playerMatchId = int(str(player['player']['id']) + str(matchId))
                player['statistics']['averageX'] = averageX
                player['statistics']['averageY'] = averageY
                player['statistics']['position'] = formation[x]
                stats[playerMatchId] = player['statistics']
    return stats


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


# with open('data/positionalStatsPrem2425.json', "w", encoding="utf-8") as f:
#   json.dump(getSeasonStatList(leagueId, seasonId), f, indent=4)


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

# statCategories.remove('goals')

positions = ["RB", "LB", "CB", "DM", "RM", "LM", "CM", "RW", "LW", "AM", "ST"]

formations = [('CB', 'CB', 'CB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'DM', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'DM', 'LM', 'AM', 'RW', 'ST', 'LW'),
              ('CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'RW', 'AM', 'LW', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'LM', 'AM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'RW', 'ST', 'LW'),
              ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'LM', 'RW', 'ST', 'LW'),
              ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'DM', 'CM', 'CM', 'LM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'RM', 'DM', 'CM', 'CM', 'LM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'AM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'RW', 'CM', 'CM', 'LW', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'AM', 'LW', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'RW', 'ST', 'ST', 'LW'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'ST', 'ST', 'LW'),
              ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'AM', 'AM', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'RW', 'ST', 'LW'),
              ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'AM', 'RW', 'ST', 'LW'),
              ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'CM', 'RW', 'ST', 'LW'),
              ('RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'RM', 'DM', 'DM', 'LM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST'),
              ('RB', 'CB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'ST', 'ST'),
              ('RB', 'CB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST'),
              ('RB', 'CB', 'CB', 'CB', 'LB', 'RW', 'CM', 'CM', 'LW', 'ST')]

formationsKey = {
    ('CB', 'CB', 'CB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST'): '3-1-4-2',
    ('CB', 'CB', 'CB', 'DM', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST'): '3-2-4-1',
    ('CB', 'CB', 'CB', 'RM', 'DM', 'LM', 'AM', 'RW', 'ST', 'LW'): '3-3-1-3',
    ('CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'RW', 'AM', 'LW', 'ST'): '3-3-3-1',
    ('CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST', 'ST'): '3-4-1-2',
    ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST', 'ST'): '3-4-1-2 (2)',
    ('CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'AM', 'AM', 'ST'): '3-4-2-1',
    ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'LM', 'AM', 'AM', 'ST'): '3-4-2-1 (2)',
    ('CB', 'CB', 'CB', 'RM', 'DM', 'DM', 'LM', 'RW', 'ST', 'LW'): '3-4-3',
    ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'LM', 'RW', 'ST', 'LW'): '3-4-3 (2)',
    ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'AM', 'ST'): '3-5-1-1',
    ('CB', 'CB', 'CB', 'RM', 'DM', 'CM', 'CM', 'LM', 'AM', 'ST'): '3-5-1-1 (2)',
    ('CB', 'CB', 'CB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST', 'ST'): '3-5-2',
    ('CB', 'CB', 'CB', 'RM', 'DM', 'CM', 'CM', 'LM', 'ST', 'ST'): '3-5-2 (2)',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'AM', 'ST', 'ST'): '4-1-2-1-2',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'RM', 'CM', 'CM', 'LM', 'ST'): '4-1-4-1',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'RW', 'CM', 'CM', 'LW', 'ST'): '4-1-4-1 (2)',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST'): '4-2-2-2',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'AM', 'LW', 'ST'): '4-2-3-1',
    ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'RW', 'ST', 'ST', 'LW'): '4-2-4',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'RW', 'ST', 'ST', 'LW'): '4-2-4 (2)',
    ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST'): '4-3-1-2',
    ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST'): '4-3-2-1',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'AM', 'AM', 'ST'): '4-3-2-1 (2)',
    ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'RW', 'ST', 'LW'): '4-3-3',
    ('RB', 'CB', 'CB', 'LB', 'CM', 'CM', 'AM', 'RW', 'ST', 'LW'): '4-3-3 (2)',
    ('RB', 'CB', 'CB', 'LB', 'DM', 'DM', 'CM', 'RW', 'ST', 'LW'): '4-3-3 (3)',
    ('RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'AM', 'ST'): '4-4-1-1',
    ('RB', 'CB', 'CB', 'LB', 'RM', 'DM', 'DM', 'LM', 'AM', 'ST'): '4-4-1-1 (2)',
    ('RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST', 'ST'): '4-4-2',
    ('RB', 'CB', 'CB', 'LB', 'RM', 'DM', 'DM', 'LM', 'ST', 'ST'): '4-4-2 (2)',
    ('RB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'CM', 'LM', 'ST'): '4-5-1',
    ('RB', 'CB', 'CB', 'CB', 'LB', 'CM', 'CM', 'CM', 'ST', 'ST'): '5-3-2',
    ('RB', 'CB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'ST', 'ST'): '5-3-2 (2)',
    ('RB', 'CB', 'CB', 'CB', 'LB', 'RM', 'CM', 'CM', 'LM', 'ST'): '5-4-1',
    ('RB', 'CB', 'CB', 'CB', 'LB', 'RW', 'CM', 'CM', 'LW', 'ST'): '5-4-1 (2)'
}

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

model, X = MLModel()

TeamA = input("\nName the home team: ")
homeTeam = getFullTeamName(TeamA)
TeamB = input("\nName the away team: ")
awayTeam = getFullTeamName(TeamB)

searches = jsonRequest(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=0")['results']
if safe_request(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=1"):
    searches.extend(jsonRequest(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=1")['results'])
if safe_request(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=2"):
    searches.extend(jsonRequest(f"http://www.sofascore.com/api/v1/search/events?q={homeTeam}%20{awayTeam}&page=2")['results'])
matchList = []

for results in searches:
    result = results['entity']
    epoch = result['startTimestamp']
    dt_object = datetime.datetime.fromtimestamp(epoch)
    human_readable_time = dt_object.strftime('%d-%m-%Y')
    if result['homeTeam']['name'] == homeTeam and result['awayTeam']['name'] == awayTeam and epoch > 1656630000:
        matchList.append((result['name'], result['id'], human_readable_time, epoch))

matchList = sorted(matchList, key=lambda x: x[3], reverse=True)

for x in range(len(matchList)):
    print(f"{x + 1}. {matchList[x][2]}")
choice = int(input("\nWhich number fixture would you like to select? ")) - 1

matchId = matchList[choice][1]

lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups")
averagePositions = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/average-positions")

homeLineups = lineups['home']['players']
awayLineups = lineups['away']['players']

homePlayerStats = []
homePlayerKey = {}

count = 0
for player in homeLineups:
    if player['position'] != 'G' and count < 10:
        count += 1
        for average in averagePositions['home']:
            if average['player']['id'] == player['player']['id']:
                player['statistics']['averageX'] = average['averageX']
                player['statistics']['averageY'] = average['averageY']
                homePlayerKey[count] = player['player']['name']
        homePlayerStats.append(player['statistics'])

awayPlayerStats = []
awayPlayerKey = {}

count = 0
for player in awayLineups:
    if player['position'] != 'G' and count < 10:
        count += 1
        for average in averagePositions['away']:
            if average['player']['id'] == player['player']['id']:
                player['statistics']['averageX'] = average['averageX']
                player['statistics']['averageY'] = average['averageY']
                awayPlayerKey[count] = player['player']['name']
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


def show_player_probabilities(player_index, prob_matrix, key, home=True):
    # Invert the key dictionary so we can map index → position name
    index_to_pos = {v: k for k, v in key.items()}

    # Get that player’s probabilities
    probs = prob_matrix[player_index]

    # Convert to a sorted DataFrame for readability
    df = pd.DataFrame({
        "Position": [index_to_pos[i] for i in range(len(probs))],
        "Probability": [float(f"{p:.4f}") for p in probs]
    }).sort_values(by="Probability", ascending=False).reset_index(drop=True)
    if home:
        print(f"\nProbabilities for Player {homePlayerKey[player_index + 1]}:")
    else:
        print(f"\nProbabilities for Player {awayPlayerKey[player_index + 1]}:")
    print(df.to_string(index=False))
    return df


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
            (p, slots[s], float(prob_matrix[p, positions.index(slots[s])]))
            for p, s in all_pairs
        ]
        return total_prob, mapping

    return total_prob


def teamEval(xProb, teamName, home=True):
    formation_scores = []
    for f in formations:
        score = formation_likelihood(xProb, f, positions)
        formation_scores.append((formationsKey[f], score))

    # Sort by likelihood
    formation_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{teamName}:")

    print("\nMost likely formations:")
    for f, s in formation_scores[:5]:
        print(f"{f}: likelihood = {round_sig(s)}")

    # Get best formation name
    best_formation_name = formation_scores[0][0]

    # Find the corresponding formation tuple
    best_f = None
    for k, v in formationsKey.items():
        if v == best_formation_name:
            best_f = k
            break

    # Get optimal mapping for that formation
    _, mapping = formation_likelihood(xProb, best_f, positions, return_mapping=True)

    mapping.sort(key=lambda x: x[0])

    print(f"\nOptimal player-to-position mapping for {teamName}:\n")
    if home:
        for player, position, prob in mapping:
            print(f"Player {player + 1} ({homePlayerKey[player + 1]}) → {position} ({prob:.4f})")
    else:
        for player, position, prob in mapping:
            print(f"Player {player + 1} ({awayPlayerKey[player + 1]}) → {position} ({prob:.4f})")


teamEval(homeProb, homeTeam)
teamEval(awayProb, awayTeam, home=False)

answer = input("Would you like to see a player's probability distribution? (y/n) ")
if answer == "y":
    homeAway = input("Home or away? (h/a) ")
    player_index = int(input("\nWhich player number would you like to see? ")) - 1
    if homeAway == "h":
        show_player_probabilities(player_index, homeProb, key)
    elif homeAway == "a":
        show_player_probabilities(player_index, awayProb, key, home=False)
