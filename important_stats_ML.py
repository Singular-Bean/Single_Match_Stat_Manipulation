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
import matplotlib.pyplot as plt


def jsonRequest(url, cache_ttl=None):
    if cache_ttl is None:
        resp = safe_request(url)  # uses default CACHE_TTL
    else:
        cache_ttl = cache_ttl * 3600
        resp = safe_request(url, cache_ttl=cache_ttl)
    return resp.json()


def leagueId():
    response = input("Enter League Name: ")
    return jsonRequest(f"http://www.sofascore.com/api/v1/search/unique-tournaments?q={response}&page=0", cache_ttl=0)[
        'results'][0][
        'entity']['id']


def statsCheckSeasonIds(leagueid):
    def statsValid(matchId):
        response = safe_request(f"http://www.sofascore.com/api/v1/event/{matchId}/statistics", cache_ttl=0)
        if response.status_code == 200:
            return True

    seasons = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueid}/seasons", cache_ttl=24 * 7)[
        'seasons']

    seasonIds = []

    for season in seasons:
        seasonIds.append(season['id'])

    validSeasons = []

    for seasonId in seasonIds:
        firstGame = jsonRequest(
            f"http://www.sofascore.com/api/v1/unique-tournament/{leagueid}/season/{seasonId}/events/round/1",
            cache_ttl=24)

        if 'events' in firstGame:
            firstGame = firstGame['events'][0]

            if statsValid(firstGame['id']):
                validSeasons.append(seasonId)

    validSeasonNames = []

    for season in seasons:
        if season['id'] in validSeasons:
            validSeasonNames.append(season)
    return validSeasonNames


def getMD1Stats(seasonId, leagueId):
    round1 = \
        jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/1",
                    cache_ttl=0)[
            'events']
    round2 = \
        jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/events/round/2",
                    cache_ttl=0)[
            'events']
    bothRounds = round1 + round2
    statCategories = []

    for x in bothRounds:
        stats = jsonRequest(f"http://www.sofascore.com/api/v1/event/{x['id']}/statistics", cache_ttl=0)
        if 'statistics' in stats:
            stats = stats['statistics'][0]['groups']
            for group in stats:
                items = group['statisticsItems']
                for stat in items:
                    if stat['name'] not in statCategories:
                        statCategories.append(stat['name'])
    print(len(statCategories))
    return statCategories


def getAllMatchIds(leagueId, seasons):
    def getRounds(leagueId, seasonId):
        gamesPlayed = []
        standings = \
        jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{seasonId}/standings/total")[
            'standings'][0]['rows']
        for x in standings:
            gamesPlayed.append(x['matches'])
        return max(gamesPlayed)

    total = []
    matchIdToResult = []
    matchIdHalfTimeScore = []
    for season in seasons:
        rounds = getRounds(leagueId, season)
        for round in range(rounds):
            roundEvents = jsonRequest(
                f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/season/{season}/events/round/{round + 1}",
                cache_ttl=24 * 14)[
                'events']
            for event in roundEvents:
                total.append(event['id'])
                if 'winnerCode' in event:
                    matchIdToResult.append((event['id'], event['winnerCode'] - 1))
                if 'homeScore' in event and 'awayScore' in event:
                    if 'period1' in event['homeScore'] and 'period1' in event['awayScore']:
                        matchIdHalfTimeScore.append(
                            (event['id'], (event['homeScore']['period1'], event['awayScore']['period1'])))
    return total, matchIdToResult, matchIdHalfTimeScore


import random
import copy


def randomFlipMatches(data, flip_prob=0.5, random_state=42):
    random.seed(random_state)
    flipped_data = copy.deepcopy(data)
    keys = list(flipped_data.keys())

    for key in keys:
        if random.random() < flip_prob:
            match = flipped_data[key]
            # Swap all two-element stats
            for stat, val in match.items():
                if stat == "Match result":
                    continue
                if isinstance(val, list) and len(val) == 2:
                    match[stat] = [val[1], val[0]]

            # Flip result (0 ↔ 1), leave draws (2) unchanged
            if match["Match result"] == 0:
                match["Match result"] = 1
            elif match["Match result"] == 1:
                match["Match result"] = 0

    return flipped_data


def differenceConvert(data):
    copyData = copy.deepcopy(data)
    keys = list(copyData.keys())

    for key in keys:
        for category in categories:
            copyData[key][category] = copyData[key][category][0] - copyData[key][category][1]

    return copyData


leagueId = 17

# leagueId = leagueId()
# seasonsWithStats = statsCheckSeasonIds(leagueId)


totalStats = []

"""
for y in seasonsWithStats:
    uniqueStats = []
    seasonStats = getMD1Stats(y['id'], leagueId)
    for w in seasonStats:
        if w not in totalStats:
            totalStats.append(w)
            uniqueStats.append(w)
    print(f"Stats introduced in {y['name']}: ", uniqueStats)
"""

usableSeasons = [76986, 61627, 52186, 41886]

categories = ['Ball possession', 'Expected goals', 'Big chances', 'Total shots', 'Goalkeeper saves', 'Corner kicks',
              'Fouls', 'Passes', 'Tackles', 'Free kicks', 'Yellow cards', 'Shots on target', 'Hit woodwork',
              'Shots off target', 'Blocked shots', 'Shots inside box', 'Shots outside box', 'Big chances scored',
              'Big chances missed', 'Through balls', 'Fouled in final third', 'Offsides', 'Accurate passes',
              'Throw-ins', 'Final third entries', 'Long balls', 'Crosses', 'Duels', 'Dispossessed', 'Ground duels',
              'Aerial duels', 'Dribbles', 'Tackles won', 'Total tackles', 'Interceptions', 'Clearances',
              'Total saves',
              'Goals prevented', 'Goal kicks', 'Red cards', 'Recoveries']


def createJsonData(leagueId, usableSeasons, filePath, halTime):
    if halTime == 1:
        rawIdList, matchIdToResult, matchIdHalfTimeScore = getAllMatchIds(leagueId, usableSeasons)

        filteredIdList = []

        length = round(len(rawIdList) / 10)

        for id in rawIdList:
            place = rawIdList.index(id)
            if place % length == 0:
                print(f"{round((place * 100) / (length * 10))}% complete")
            if safe_request(f"http://www.sofascore.com/api/v1/event/{id}/statistics", cache_ttl=0).status_code == 200:
                filteredIdList.append(id)
        print(len(filteredIdList))
        print(filteredIdList)

        statCategories = []

        preDict = {}

        for id in filteredIdList:
            preDict[id] = {}
            for cat in categories:
                preDict[id][cat] = (0, 0)
            matchStats = \
            jsonRequest(f"http://www.sofascore.com/api/v1/event/{id}/statistics", cache_ttl=0)['statistics'][1][
                'groups']
            for group in matchStats:
                items = group['statisticsItems']
                for stat in items:
                    if stat['name'] not in statCategories:
                        statCategories.append(stat['name'])
                    if stat['name'] in categories:
                        preDict[id][stat['name']] = (stat['homeValue'], stat['awayValue'])

        for n in matchIdToResult:
            if n[0] in preDict:
                preDict[n[0]]['Match result'] = n[1]

        for m in matchIdHalfTimeScore:
            if m[0] in preDict:
                preDict[m[0]]['Half time home score'] = m[1][0]
                preDict[m[0]]['Half time away score'] = m[1][1]
                preDict[m[0]]['Half time result'] = m[1][0] - m[1][1]
    else:
        rawIdList, matchIdToResult, matchIdHalfTimeScore = getAllMatchIds(leagueId, usableSeasons)

        filteredIdList = []

        for id in rawIdList:
            if safe_request(f"http://www.sofascore.com/api/v1/event/{id}/statistics", cache_ttl=0).status_code == 200:
                filteredIdList.append(id)
        print(len(filteredIdList))
        print(filteredIdList)

        statCategories = []

        preDict = {}

        for id in filteredIdList:
            preDict[id] = {}
            for cat in categories:
                preDict[id][cat] = (0, 0)
            matchStats = \
            jsonRequest(f"http://www.sofascore.com/api/v1/event/{id}/statistics", cache_ttl=0)['statistics'][0][
                'groups']
            for group in matchStats:
                items = group['statisticsItems']
                for stat in items:
                    if stat['name'] not in statCategories:
                        statCategories.append(stat['name'])
                    if stat['name'] in categories:
                        preDict[id][stat['name']] = (stat['homeValue'], stat['awayValue'])


        for n in matchIdToResult:
            if n[0] in preDict:
                preDict[n[0]]['Match result'] = n[1]

        for match_id in list(preDict.keys()):
            stats = preDict[match_id]
            if 'Match result' not in stats:
                # Remove the item if the key is missing
                preDict.pop(match_id)

    with open(filePath, "w", encoding="utf-8") as f:
        json.dump(preDict, f, indent=4)


#createJsonData(leagueId, usableSeasons, "data/matchStats.json", 0)

file_path = "data/matchStats.json"

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

data = differenceConvert(randomFlipMatches(data))
# data = differenceConvert(data)

df = pd.DataFrame(data).T

X = df.drop(columns=['Match result', 'Big chances scored', 'Big chances missed', 'Goalkeeper saves', 'Goals prevented', 'Total saves',
                     'Total tackles'])

y = df['Match result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = XGBClassifier(
    objective='multi:softprob',  # for multiclass probability output
    num_class=3,  # 3 classes: home win / away win / draw
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict probabilities
y_prob = model.predict_proba(X_test)
print("Log loss:", log_loss(y_test, y_prob))

importanceModel = XGBClassifier(
    objective='multi:softprob',  # for multiclass probability output
    num_class=3,  # 3 classes: home win / away win / draw
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='mlogloss',
    random_state=42
)

importanceModel.fit(X, y)

importances = importanceModel.feature_importances_
importance_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": importances})
    .sort_values("Importance", ascending=False)
)
print(importance_df)

y_true = np.array(y_test)

y_prob = np.array(y_prob)

brier_class0 = brier_score_loss(y_true == 0, y_prob[:, 0])
brier_class1 = brier_score_loss(y_true == 1, y_prob[:, 1])
brier_class2 = brier_score_loss(y_true == 2, y_prob[:, 2])

# Or average them to get overall score
brier_score = (brier_class0 + brier_class1 + brier_class2) / 3

idsWithProbs = []

y_prob = list(y_prob)
clean_list = [arr.tolist() for arr in y_prob]

for i in clean_list:
    index = clean_list.index(i)
    confidentMatchId = y_train.keys()[index]
    if max(i) > 0.95:
        idsWithProbs.append((confidentMatchId, i))

print("\n--- Running SHAP on Home Win Probability ---")

def home_win_predict(X_input):
    return importanceModel.predict_proba(X_input)[:, 0]

explainer = shap.Explainer(home_win_predict, X)
shap_values_home = explainer(X_test)

# Plot ALL features (no truncation) and increase image size
shap.summary_plot(
    shap_values_home.values,
    X_test,
    show=True,
    plot_size=(10, 20),   # increase figure size
    max_display=len(X_test.columns)  # show all features
)