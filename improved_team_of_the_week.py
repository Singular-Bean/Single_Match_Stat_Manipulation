import pandas as pd
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from functions_store import *


def getCompetitionId():
    data = \
        jsonRequest(f"http://www.sofascore.com/api/v1/search/unique-tournaments?q={input("Competition name: ")}&page=0",
                    cache_ttl=0)['results']
    count = 0
    for i in data:
        if count < 10 and 'gender' in i['entity']:
            count += 1
            print(f"{count}. {i['entity']['name']} ({i['entity']['gender']})")
    choice = input("Which number would you like to choose? ")
    return data[int(choice) - 1]['entity']['id']


def getSeasonIds(leagueId):
    validList = []
    data = jsonRequest(f"http://www.sofascore.com/api/v1/unique-tournament/{leagueId}/seasons", cache_ttl=24 * 7)[
        'seasons']
    for season in data:
        yearName = season['year']
        years = yearName.split('/')
        first = int(years[0])
        if first > 2000:
            first -= 2000
        elif first > 1900:
            first -= 1900
        if 90 > first > 21:
            validList.append((season['id'], season['name']))
    return validList


def generate_team_combinations(players, team_size=10):
    n = len(players)

    for combo_indices in combinations(range(n), team_size):
        # Create the key: concatenate all indexes, zero-padded to 2 digits if needed
        key = ''.join(f"{i:02d}" for i in combo_indices)

        # Retrieve the corresponding player tuples
        combo = [players[i] for i in combo_indices]

        yield key, combo


positions = ["RB", "LB", "CB", "DM", "RM", "LM", "CM", "RW", "LW", "AM", "ST"]


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
    ('RB', 'CB', 'CB', 'LB', 'DM', 'CM', 'CM', 'CM', 'ST', 'ST'): '4-1-3-2',
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

formations = formationsKey.keys()

inverseFormationsKey = {v: k for k, v in formationsKey.items()}

competitionId = getCompetitionId()
seasonIdsAndYears = getSeasonIds(competitionId)

for season in seasonIdsAndYears:
    print(f"{seasonIdsAndYears.index(season) + 1}. {season[1]}")
choice = input("Which number would you like to choose? ")
chosenSeasonId = seasonIdsAndYears[int(choice) - 1][0]

roundInfo = jsonRequest(
    f"http://www.sofascore.com/api/v1/unique-tournament/{competitionId}/season/{chosenSeasonId}/rounds", cache_ttl=24*3)
currentRound = roundInfo['currentRound']
rounds = roundInfo['rounds']

if 'name' in currentRound:
    currentRound = currentRound['name']
else:
    currentRound = currentRound['round']

switch = True
for round in rounds:
    if switch:
        if 'name' in round:
            print(f"{rounds.index(round) + 1}. {round['name']}")
            if round['name'] == currentRound:
                switch = False
        else:
            print(f"{rounds.index(round) + 1}. Round {round['round']}")
            if round['round'] == currentRound:
                switch = False
roundChoice = int(input("Which round would you like to choose? "))
# roundchoice = 0
round = rounds[roundChoice - 1]
if 'prefix' in round:
    round = (round['round'], round['slug'], round['prefix'])
    roundGames = jsonRequest(
        f"http://www.sofascore.com/api/v1/unique-tournament/{competitionId}/season/{chosenSeasonId}/events/round/{round[0]}/slug/{round[1]}/prefix/{round[2]}",
        cache_ttl=24 * 7)[
        'events']
elif 'slug' in round:
    round = (round['round'], round['slug'])
    roundGames = jsonRequest(
        f"http://www.sofascore.com/api/v1/unique-tournament/{competitionId}/season/{chosenSeasonId}/events/round/{round[0]}/slug/{round[1]}",
        cache_ttl=24 * 7)[
        'events']
else:
    round = round['round']
    roundGames = jsonRequest(
        f"http://www.sofascore.com/api/v1/unique-tournament/{competitionId}/season/{chosenSeasonId}/events/round/{round}",
        cache_ttl=24 * 7)[
        'events']

if roundGames[-1]['status']['code'] != 100:
    raise RuntimeError("Round has not been completed, choose a different one.")

roundIds = []

for game in roundGames:
    roundIds.append(game['id'])

stats = {}

idsToNames = {}

for matchId in roundIds:
    lineups = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/lineups", cache_ttl=0)
    averagePositions = jsonRequest(f"http://www.sofascore.com/api/v1/event/{matchId}/average-positions", cache_ttl=0)
    for x in lineups['home']['players']:
        x = lineups['home']['players'].index(x)
        if x < 11:
            player = lineups['home']['players'][x]
            for y in averagePositions['home']:
                if player['player']['id'] == y['player']['id']:
                    averageX = y['averageX']
                    averageY = y['averageY']
            playerId = player['player']['id']
            index = int(f"{playerId}{matchId}")
            idsToNames[index] = player['player']['name']
            formation = getFormation('home')
            if 'statistics' in player:
                player['statistics']['averageX'] = averageX
                player['statistics']['averageY'] = averageY
                player['statistics']['position'] = formation[x]
                stats[index] = player['statistics']
    for x in lineups['away']['players']:
        x = lineups['away']['players'].index(x)
        if x < 11:
            player = lineups['away']['players'][x]
            for y in averagePositions['away']:
                if player['player']['id'] == y['player']['id']:
                    averageX = y['averageX']
                    averageY = y['averageY']
            playerId = player['player']['id']
            index = int(f"{playerId}{matchId}")
            idsToNames[index] = player['player']['name']
            formation = getFormation('away')
            if 'statistics' in player:
                player['statistics']['averageX'] = averageX
                player['statistics']['averageY'] = averageY
                player['statistics']['position'] = formation[x]
                stats[index] = player['statistics']

df = pd.DataFrame(stats).T.infer_objects(copy=False).fillna(0)

goalies = df[df['position'] == 'GK']

df = df[df['position'] != 'GK']

df = df[df['rating'] > 0]

model, X_train = MLModel()

statsColumns = list(X_train.columns)

X = df.reindex(columns=statsColumns, fill_value=0)

# Force the live data to be numeric, just like you did with the training data
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

y_prob = list(model.predict_proba(X))

y_ratings = list(X['rating'])

y_indices = [np.argmax(a) for a in y_prob]

y_idmatches = df.index.tolist()

goalieRatings = list(goalies['rating'])

if 'goalsPrevented' in goalies.columns:
    goalieGoalsPrevented = list(goalies['goalsPrevented'])
else:
    goalieGoalsPrevented = []
    for x in range(len(goalies)):
        goalieGoalsPrevented.append(0)

goalieIds = list(goalies.T.keys())

bestGoalie = (0, 0, 0)

for goalie in range(len(goalies)):
    goalieRating = goalieRatings[goalie]
    goalieGoals = goalieGoalsPrevented[goalie]
    goalieId = goalieIds[goalie]

    if goalieRating > bestGoalie[1]:
        bestGoalie = (goalieId, goalieRating, goalieGoals)
    elif goalieRating == bestGoalie[1]:
        if goalieGoals > bestGoalie[2]:
            bestGoalie = (goalieId, goalieRating, goalieGoals)

RB = []
LB = []
CB = []
DM = []
RM = []
LM = []
CM = []
RW = []
LW = []
AM = []
ST = []

for i in range(0, len(y_idmatches)):
    if y_indices[i] == 0:
        RB.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 1:
        LB.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 2:
        CB.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 3:
        DM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 4:
        RM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 5:
        LM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 6:
        CM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 7:
        RW.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 8:
        LW.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 9:
        AM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 10:
        ST.append((y_idmatches[i], y_prob[i], y_ratings[i]))

RB.sort(key=lambda x: x[2], reverse=True)
topRB = RB[:3]

LB.sort(key=lambda x: x[2], reverse=True)
topLB = LB[:3]

CB.sort(key=lambda x: x[2], reverse=True)
topCB = CB[:4]

DM.sort(key=lambda x: x[2], reverse=True)
topDM = DM[:3]

RM.sort(key=lambda x: x[2], reverse=True)
topRM = RM[:3]

LM.sort(key=lambda x: x[2], reverse=True)
topLM = LM[:3]

CM.sort(key=lambda x: x[2], reverse=True)
topCM = CM[:4]

RW.sort(key=lambda x: x[2], reverse=True)
topRW = RW[:3]

LW.sort(key=lambda x: x[2], reverse=True)
topLW = LW[:3]

AM.sort(key=lambda x: x[2], reverse=True)
topAM = AM[:3]

ST.sort(key=lambda x: x[2], reverse=True)
topST = ST[:3]

tops = topRB + topLB + topCB + topDM + topRM + topLM + topCM + topRW + topLW + topAM + topST

ratingKey = {}
topsRatings = []

for player in tops:
    topsRatings.append(player[2])

newRatings = scale_ratings_to_10(topsRatings, give=0)

for num in range(0, len(newRatings)):
    ratingKey[topsRatings[num]] = newRatings[num]

highestRatedSquadScore = -float('inf')
highestRatedSquad = None

numPlayers = len(tops)

# Iterate only through the 37 formations, eliminating the combinations loop
for f in formations:
    numSlots = len(f)

    # Initialize the cost matrix (Rows: All available players, Columns: 10 formation slots)
    costMatrix = np.zeros((numPlayers, numSlots))

    for i in range(numPlayers):
        playerId = tops[i][0]
        probArray = tops[i][1]
        rawRating = tops[i][2]

        playerRating = ratingKey[rawRating]
        logRating = math.log(playerRating) if playerRating > 0 else -999

        for j in range(numSlots):
            positionName = f[j]
            positionIndex = positions.index(positionName)
            prob = probArray[positionIndex]

            # Use a small epsilon to prevent math domain errors on log(0)
            logProb = math.log(prob) if prob > 1e-6 else -999

            # The algorithm minimizes cost, so we negate the sum to maximize the score
            costMatrix[i, j] = -(logRating + logProb)

    # The solver automatically picks the best 10 players and their optimal slots
    rowInd, colInd = linear_sum_assignment(costMatrix)

    totalRating = 1.0
    totalProb = 1.0
    currentTeam = []
    currentMapping = []

    for idx in range(numSlots):
        playerIdx = rowInd[idx]
        slotIdx = colInd[idx]

        probArray = tops[playerIdx][1]
        rawRating = tops[playerIdx][2]
        positionName = f[slotIdx]
        positionIndex = positions.index(positionName)

        playerRating = ratingKey[rawRating]
        prob = probArray[positionIndex]

        totalRating *= playerRating
        totalProb *= prob

        currentTeam.append(tops[playerIdx])
        currentMapping.append((f"Player {playerIdx + 1}", positionName, float(prob)))

    totalRatingScore = totalRating * totalProb

    # Track the highest scoring formation
    if totalRatingScore > highestRatedSquadScore:
        highestRatedSquadScore = totalRatingScore

        # Format the key to match your expected tuple structure downstream
        comboIndices = sorted(rowInd)
        comboKey = ''.join(f"{idx:02d}" for idx in comboIndices)

        highestRatedSquad = (comboKey, currentTeam, totalRatingScore, formationsKey[f], currentMapping)

for player in range(len(highestRatedSquad[1])):
    temp = list(highestRatedSquad[1][player])
    temp[0] = idsToNames[temp[0]]
    highestRatedSquad[1][player] = tuple(temp)

selectPlayers = highestRatedSquad[1]
order = highestRatedSquad[4]
formation = inverseFormationsKey[highestRatedSquad[3]]

for x in range(len(order)):
    order[x] = list(order[x])

for y in range(len(order)):
    order[y].append(selectPlayers[y][2])
    order[y][0] = selectPlayers[y][0]

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

order.sort(key=lambda x: key[x[1]], reverse=True)

print(f"\nFormation: {highestRatedSquad[3]}")

if 'ST' in formation:
    print("\nStrikers:")
    for player in order:
        if player[1] == 'ST':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'LW' in formation:
    print("\nLeft Wingers:")
    for player in order:
        if player[1] == 'LW':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'AM' in formation:
    print("\nAttacking Midfielders:")
    for player in order:
        if player[1] == 'AM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'RW' in formation:
    print("\nRight Wingers:")
    for player in order:
        if player[1] == 'RW':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'LM' in formation:
    print("\nLeft Midfielders:")
    for player in order:
        if player[1] == 'LM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'CM' in formation:
    print("\nCentral Midfielders:")
    for player in order:
        if player[1] == 'CM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'RM' in formation:
    print("\nRight Midfielders:")
    for player in order:
        if player[1] == 'RM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'DM' in formation:
    print("\nDefensive Midfielders:")
    for player in order:
        if player[1] == 'DM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'LB' in formation:
    print("\nLeft Backs:")
    for player in order:
        if player[1] == 'LB':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'CB' in formation:
    print("\nCenter Backs:")
    for player in order:
        if player[1] == 'CB':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'RB' in formation:
    print("\nRight Backs:")
    for player in order:
        if player[1] == 'RB':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")

print("\nGoalkeeper:")
print(f"{idsToNames[bestGoalie[0]]} (Rating: {bestGoalie[1]})")
