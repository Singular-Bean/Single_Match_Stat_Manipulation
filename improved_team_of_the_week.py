import pandas as pd
import numpy as np
from itertools import combinations
from functions_store import jsonRequest, MLModel, formation_likelihood, round_sig, scale_ratings_to_10


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


positions = ["FB", "CB", "DM", "WM", "CM", "W", "AM", "ST"]

formations = [('CB', 'CB', 'CB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'DM', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'DM', 'WM', 'AM', 'W', 'ST', 'W'),
              ('CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'W', 'AM', 'W', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'WM', 'AM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'WM', 'AM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'W', 'ST', 'W'),
              ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'WM', 'W', 'ST', 'W'),
              ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'DM', 'CM', 'CM', 'WM', 'AM', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST', 'ST'),
              ('CB', 'CB', 'CB', 'WM', 'DM', 'CM', 'CM', 'WM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'CM', 'CM', 'AM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'W', 'CM', 'CM', 'W', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'W', 'AM', 'W', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'W', 'ST', 'ST', 'W'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'W', 'ST', 'ST', 'W'),
              ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'CM', 'CM', 'AM', 'AM', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'W', 'ST', 'W'),
              ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'AM', 'W', 'ST', 'W'),
              ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'CM', 'W', 'ST', 'W'),
              ('FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'AM', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'WM', 'DM', 'DM', 'WM', 'AM', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'WM', 'DM', 'DM', 'WM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST'),
              ('FB', 'CB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'CB', 'FB', 'DM', 'CM', 'CM', 'ST', 'ST'),
              ('FB', 'CB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST'),
              ('FB', 'CB', 'CB', 'CB', 'FB', 'W', 'CM', 'CM', 'W', 'ST')]

formationsKey = {
    ('CB', 'CB', 'CB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST') : '3-1-4-2',
    ('CB', 'CB', 'CB', 'DM', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST') : '3-2-4-1',
    ('CB', 'CB', 'CB', 'WM', 'DM', 'WM', 'AM', 'W', 'ST', 'W') : '3-3-1-3',
    ('CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'W', 'AM', 'W', 'ST') : '3-3-3-1',
    ('CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'ST', 'ST') : '3-4-1-2',
    ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'WM', 'AM', 'ST', 'ST') : '3-4-1-2 (2)',
    ('CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'AM', 'ST') : '3-4-2-1',
    ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'WM', 'AM', 'AM', 'ST') : '3-4-2-1 (2)',
    ('CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'W', 'ST', 'W') : '3-4-3',
    ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'WM', 'W', 'ST', 'W') : '3-4-3 (2)',
    ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'AM', 'ST') : '3-5-1-1',
    ('CB', 'CB', 'CB', 'WM', 'DM', 'CM', 'CM', 'WM', 'AM', 'ST') : '3-5-1-1 (2)',
    ('CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST', 'ST') : '3-5-2',
    ('CB', 'CB', 'CB', 'WM', 'DM', 'CM', 'CM', 'WM', 'ST', 'ST') : '3-5-2 (2)',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'CM', 'CM', 'AM', 'ST', 'ST') : '4-1-2-1-2',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST') : '4-1-4-1',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'W', 'CM', 'CM', 'W', 'ST') : '4-1-4-1 (2)',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST') : '4-2-2-2',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'W', 'AM', 'W', 'ST') : '4-2-3-1',
    ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'W', 'ST', 'ST', 'W') : '4-2-4',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'W', 'ST', 'ST', 'W') : '4-2-4 (2)',
    ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST') : '4-3-1-2',
    ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST') : '4-3-2-1',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'CM', 'CM', 'AM', 'AM', 'ST') : '4-3-2-1 (2)',
    ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'W', 'ST', 'W'): '4-3-3',
    ('FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'AM', 'W', 'ST', 'W'): '4-3-3 (2)',
    ('FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'CM', 'W', 'ST', 'W'): '4-3-3 (3)',
    ('FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'AM', 'ST'): '4-4-1-1',
    ('FB', 'CB', 'CB', 'FB', 'WM', 'DM', 'DM', 'WM', 'AM', 'ST'): '4-4-1-1 (2)',
    ('FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST'): '4-4-2',
    ('FB', 'CB', 'CB', 'FB', 'WM', 'DM', 'DM', 'WM', 'ST', 'ST'): '4-4-2 (2)',
    ('FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST'): '4-5-1',
    ('FB', 'CB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'ST', 'ST'): '5-3-2',
    ('FB', 'CB', 'CB', 'CB', 'FB', 'DM', 'CM', 'CM', 'ST', 'ST'): '5-3-2 (2)',
    ('FB', 'CB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST'): '5-4-1',
    ('FB', 'CB', 'CB', 'CB', 'FB', 'W', 'CM', 'CM', 'W', 'ST'): '5-4-1 (2)'
}

inverseFormationsKey = {v: k for k, v in formationsKey.items()}


competitionId = getCompetitionId()
seasonIdsAndYears = getSeasonIds(competitionId)

for season in seasonIdsAndYears:
    print(f"{seasonIdsAndYears.index(season) + 1}. {season[1]}")
choice = input("Which number would you like to choose? ")
chosenSeasonId = seasonIdsAndYears[int(choice) - 1][0]

roundInfo = jsonRequest(
    f"http://www.sofascore.com/api/v1/unique-tournament/{competitionId}/season/{chosenSeasonId}/rounds", cache_ttl=24)
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
            idsToNames[player['player']['id']] = player['player']['name']
            formation = []
            if lineups['home']['formation'] == '3-1-4-2':
                formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST']
            elif lineups['home']['formation'] == '3-2-4-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST']
            elif lineups['home']['formation'] == '3-3-1-3':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'WM', 'AM', 'W', 'ST', 'W']
            elif lineups['home']['formation'] == '3-3-3-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'W', 'AM', 'W', 'ST']
            elif lineups['home']['formation'] == '3-4-1-2':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'ST', 'ST']
            elif lineups['home']['formation'] == '3-4-2-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'AM', 'ST']
            elif lineups['home']['formation'] == '3-4-3':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'W', 'ST', 'W']
            elif lineups['home']['formation'] == '3-5-1-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'AM', 'ST']
            elif lineups['home']['formation'] == '3-5-2':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST', 'ST']
            elif lineups['home']['formation'] == '4-1-4-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST']
            elif lineups['home']['formation'] == '4-2-2-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST']
            elif lineups['home']['formation'] == '4-2-3-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'W', 'AM', 'W', 'ST']
            elif lineups['home']['formation'] == '4-2-4':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'W', 'ST', 'ST', 'W']
            elif lineups['home']['formation'] == '4-3-1-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST']
            elif lineups['home']['formation'] == '4-3-2-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST']
            elif lineups['home']['formation'] == '4-3-3':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'W', 'ST', 'W']
            elif lineups['home']['formation'] == '4-4-1-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'AM', 'ST']
            elif lineups['home']['formation'] == '4-4-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST']
            elif lineups['home']['formation'] == '4-5-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST']
            elif lineups['home']['formation'] == '5-3-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'ST', 'ST']
            elif lineups['home']['formation'] == '5-4-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST']
            playerId = player['player']['id']
            if 'statistics' in player:
                player['statistics']['averageX'] = averageX
                player['statistics']['averageY'] = averageY
                player['statistics']['position'] = formation[x]
                stats[playerId] = player['statistics']
    for x in lineups['away']['players']:
        x = lineups['away']['players'].index(x)
        if x < 11:
            player = lineups['away']['players'][x]
            for y in averagePositions['away']:
                if player['player']['id'] == y['player']['id']:
                    averageX = y['averageX']
                    averageY = y['averageY']
            idsToNames[player['player']['id']] = player['player']['name']
            formation = []
            if lineups['away']['formation'] == '3-1-4-2':
                formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST']
            elif lineups['away']['formation'] == '3-2-4-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'DM', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST']
            elif lineups['away']['formation'] == '3-3-1-3':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'WM', 'AM', 'W', 'ST', 'W']
            elif lineups['away']['formation'] == '3-3-3-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'CM', 'W', 'AM', 'W', 'ST']
            elif lineups['away']['formation'] == '3-4-1-2':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'ST', 'ST']
            elif lineups['away']['formation'] == '3-4-2-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'AM', 'AM', 'ST']
            elif lineups['away']['formation'] == '3-4-3':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'DM', 'DM', 'WM', 'W', 'ST', 'W']
            elif lineups['away']['formation'] == '3-5-1-1':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'AM', 'ST']
            elif lineups['away']['formation'] == '3-5-2':
                formation = ['GK', 'CB', 'CB', 'CB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST', 'ST']
            elif lineups['away']['formation'] == '4-1-4-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'DM', 'WM', 'CM', 'CM', 'WM', 'ST']
            elif lineups['away']['formation'] == '4-2-2-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'AM', 'AM', 'ST', 'ST']
            elif lineups['away']['formation'] == '4-2-3-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'DM', 'DM', 'W', 'AM', 'W', 'ST']
            elif lineups['away']['formation'] == '4-2-4':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'W', 'ST', 'ST', 'W']
            elif lineups['away']['formation'] == '4-3-1-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'ST', 'ST']
            elif lineups['away']['formation'] == '4-3-2-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'AM', 'AM', 'ST']
            elif lineups['away']['formation'] == '4-3-3':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'W', 'ST', 'W']
            elif lineups['away']['formation'] == '4-4-1-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'AM', 'ST']
            elif lineups['away']['formation'] == '4-4-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST', 'ST']
            elif lineups['away']['formation'] == '4-5-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'CM', 'WM', 'ST']
            elif lineups['away']['formation'] == '5-3-2':
                formation = ['GK', 'FB', 'CB', 'CB', 'CB', 'FB', 'CM', 'CM', 'CM', 'ST', 'ST']
            elif lineups['away']['formation'] == '5-4-1':
                formation = ['GK', 'FB', 'CB', 'CB', 'CB', 'FB', 'WM', 'CM', 'CM', 'WM', 'ST']
            playerId = player['player']['id']
            if 'statistics' in player:
                player['statistics']['averageX'] = averageX
                player['statistics']['averageY'] = averageY
                player['statistics']['position'] = formation[x]
                stats[playerId] = player['statistics']

df = pd.DataFrame(stats).T.fillna(0)

goalies = df[df['position'] == 'GK']

df = df[df['position'] != 'GK']

df = df[df['rating'] > 0]

model, X = MLModel()

statsColumns = list(X.columns)

X = df.reindex(columns=statsColumns, fill_value=0)

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

FB = []
CB = []
DM = []
WM = []
CM = []
W  = []
AM = []
ST = []

for i in range(0, len(y_idmatches)):
    if y_indices[i] == 0:
        FB.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 1:
        CB.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 2:
        DM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 3:
        WM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 4:
        CM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 5:
        W.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 6:
        AM.append((y_idmatches[i], y_prob[i], y_ratings[i]))
    elif y_indices[i] == 7:
        ST.append((y_idmatches[i], y_prob[i], y_ratings[i]))

FB.sort(key=lambda x: x[2], reverse=True)
topFB = FB[:2]

CB.sort(key=lambda x: x[2], reverse=True)
topCB = CB[:3]

DM.sort(key=lambda x: x[2], reverse=True)
topDM = DM[:2]

WM.sort(key=lambda x: x[2], reverse=True)
topWM = WM[:2]

CM.sort(key=lambda x: x[2], reverse=True)
topCM = CM[:3]

W.sort(key=lambda x: x[2], reverse=True)
topW = W[:2]

AM.sort(key=lambda x: x[2], reverse=True)
topAM = AM[:2]

ST.sort(key=lambda x: x[2], reverse=True)
topST = ST[:2]

tops = topFB + topCB + topDM + topWM + topCM + topW + topAM + topST

ratingKey = {}
topsRatings = []

for player in tops:
    topsRatings.append(player[2])

newRatings = scale_ratings_to_10(topsRatings)

for num in range(0, len(newRatings)):
    ratingKey[topsRatings[num]] = newRatings[num]

highestRatedSquad = (0, 0, 0)

for key, team in generate_team_combinations(tops, team_size=10):
    players = []
    ratings = []
    totalRating = 1
    for t in team:
        players.append(t[1])
        ratings.append(ratingKey[t[2]])
    for r in ratings:
        totalRating *= r
    players = np.array(players)
    for f in formations:
        score, mapping = formation_likelihood(players, f, positions, return_mapping=True)
        totalRatingScore = score * totalRating
        if totalRatingScore > highestRatedSquad[2]:
            highestRatedSquad = (key, team, totalRatingScore, formationsKey[f], mapping)


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
    order[y].append(selectPlayers[int(order[y][0].split()[1])-1][2])
    order[y][0] = selectPlayers[int(order[y][0].split()[1])-1][0]

key = {
    "FB": 0,
    "CB": 1,
    "DM": 2,
    "WM": 3,
    "CM": 4,
    "W": 5,
    "AM": 6,
    "ST": 7
}

order.sort(key=lambda x: key[x[1]], reverse=True)

print(f"\nFormation: {highestRatedSquad[3]}")

if 'ST' in formation:
    print("\nStrikers:")
    for player in order:
        if player[1] == 'ST':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2]*100:.2f})")
if 'W' in formation:
    print("\nWingers:")
    for player in order:
        if player[1] == 'W':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2]*100:.2f})")
if 'AM' in formation:
    print("\nAttacking Midfielders:")
    for player in order:
        if player[1] == 'AM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2]*100:.2f})")
if 'CM' in formation:
    print("\nCentral Midfielders:")
    for player in order:
        if player[1] == 'CM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'WM' in formation:
    print("\nWide Midfielders:")
    for player in order:
        if player[1] == 'WM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'DM' in formation:
    print("\nDefensive Midfielders:")
    for player in order:
        if player[1] == 'DM':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'CB' in formation:
    print("\nCenter Backs:")
    for player in order:
        if player[1] == 'CB':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")
if 'FB' in formation:
    print("\nFull Backs:")
    for player in order:
        if player[1] == 'FB':
            print(f"{player[0]} (Rating: {player[3]}) (Positional Fit: {player[2] * 100:.2f})")

print("\nGoalkeeper:")
print(f"{idsToNames[bestGoalie[0]]} (Rating: {bestGoalie[1]})")
