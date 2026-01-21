## Important Stats ML:

This is a really simple machine learning project, exploring the "feature_importance" and SHAP explainer capabilities. The general premise was to determine what are the first things you should look at when deciding what team deserved to win a football match.

The model is trained on all match data from the premier league since the 2022/23 season as this was when xG started being introduced on SofaScore. In order to reduce the number of features, I measured the difference between stats from each team, effectively halving the number of features. For example, if one team had 5 shots on target and the other team had 7, the category would show -2, and I wouldn't need to have the value for both teams. Since taking the difference varies depending on the order of the operation, I decided to always have the home team as the one being subtracted from. This created a different problem which is that there could possibly be a home team bias built into the model. To fix this, I made it so that the program randomly flips all the signs of statistics for 50% of the matches.

After running the model, it produces the list of importance scores:
```
                  Feature  Importance
10        Shots on target    0.092230
1          Expected goals    0.084038
31             Clearances    0.046692
2             Big chances    0.039432
11           Hit woodwork    0.035946
23                Crosses    0.031785
20              Throw-ins    0.027354
22             Long balls    0.026835
26           Ground duels    0.026349
8              Free kicks    0.025938
32             Goal kicks    0.025730
3             Total shots    0.025247
14       Shots inside box    0.024777
12       Shots off target    0.024774
21    Final third entries    0.024588
29            Tackles won    0.024530
24                  Duels    0.024413
19        Accurate passes    0.024364
13          Blocked shots    0.023739
25           Dispossessed    0.023726
34             Recoveries    0.023714
6                  Passes    0.023060
28               Dribbles    0.023027
7                 Tackles    0.022621
4            Corner kicks    0.022559
27           Aerial duels    0.022060
5                   Fouls    0.021870
0         Ball possession    0.021840
15      Shots outside box    0.021447
33              Red cards    0.020942
16          Through balls    0.020535
17  Fouled in final third    0.019113
30          Interceptions    0.018576
9            Yellow cards    0.018238
18               Offsides    0.017912
```

And the SHAP explainer plot (the x axis is the change in win probability due to that feature value):
![img.png](img.png)

Some of these results are very expected (Shots on target, expected goals, etc.), but others are more interesting. For example, clearances are judged as being very important, and crosses are seen as having a negative effect on win probability.

## Estimate Position:

This program estimates the position of a player based on their single-match stats. It then does this for the entire team and uses the positional probabilities to calculate the team's most likely formation. It returns a list like this:

```
Manchester City:

Most likely formations:
4-1-4-1: likelihood = 0.00149
4-4-1-1 (2): likelihood = 0.000717
4-4-1-1: likelihood = 0.000631
4-5-1: likelihood = 0.000515
5-4-1: likelihood = 0.000146

Optimal player-to-position mapping for Manchester City:

Player 1 (Kyle Walker) → FB (0.1320)
Player 2 (Rúben Dias) → CB (0.9991)
Player 3 (Nathan Aké) → FB (0.1491)
Player 4 (Rodri) → DM (0.7346)
Player 5 (Manuel Akanji) → CB (0.9372)
Player 6 (Phil Foden) → WM (0.4815)
Player 7 (Julián Alvarez) → CM (0.3743)
Player 8 (Bernardo Silva) → CM (0.6578)
Player 9 (Jérémy Doku) → WM (0.9324)
Player 10 (Erling Haaland) → ST (0.9980)

Liverpool:

Most likely formations:
4-3-3: likelihood = 0.032299999
4-3-3 (2): likelihood = 0.0136
4-1-4-1 (2): likelihood = 0.0112
4-3-1-2: likelihood = 0.00453
4-3-2-1: likelihood = 0.00399

Optimal player-to-position mapping for Liverpool:

Player 1 (Trent Alexander-Arnold) → FB (0.5709)
Player 2 (Joël Matip) → CB (0.9991)
Player 3 (Virgil van Dijk) → CB (0.9999)
Player 4 (Konstantinos Tsimikas) → FB (0.9745)
Player 5 (Dominik Szoboszlai) → CM (0.5038)
Player 6 (Alexis Mac Allister) → CM (0.8008)
Player 7 (Curtis Jones) → CM (0.5628)
Player 8 (Mohamed Salah) → W (0.7799)
Player 9 (Darwin Núñez) → ST (0.9987)
Player 10 (Diogo Jota) → W (0.3282)
```

It then asks the user if they want to see the position probabilities for a specific player:

```
Probabilities for Player Kyle Walker:
Position  Probability
      CB       0.8664
      FB       0.1320
      WM       0.0010
      DM       0.0003
      CM       0.0002
       W       0.0000
      AM       0.0000
      ST       0.0000
```

I've found that the program generally produces good results for the individual player position probabilities, but I need to find a better way to find the most likely formations as that is quite often wrong.

## Improved Team of the Week:

I made this program because I noticed that most of the time people make a "team of the week" of the best players from a matchweek, the formations and positions of players are very unrealistic (strikers on the wing, wingers at fullback, etc.).

The program uses the same predictive model as the estimate position program to determine positional probability distributions. It applies the model to all of the player performances from the week, and then picks multiple players with the highest ratings from each position. It then takes the 18 selected players and works out all the probabilities of most likely formations with every combination of players, and multiplies them by the product of the players' ratings to get the team score. This means that the only way it would choose a really wacky formation is if the average player ratings are significantly higher because of it.

Below is an example of the output:

```
Formation: 4-3-3 (3)

Strikers:
Igor Thiago (Rating: 9.1) (Positional Fit: 95.62)

Wingers:
Alejandro Garnacho (Rating: 9.2) (Positional Fit: 83.22)
Jérémy Doku (Rating: 9.5) (Positional Fit: 76.67)

Central Midfielders:
Enzo Fernández (Rating: 7.8) (Positional Fit: 64.20)

Defensive Midfielders:
Amadou Onana (Rating: 8.2) (Positional Fit: 70.28)
Nico González (Rating: 8.3) (Positional Fit: 69.41)

Center Backs:
Michael Keane (Rating: 7.9) (Positional Fit: 99.27)
Jaydee Canvot (Rating: 7.9) (Positional Fit: 99.14)

Full Backs:
Tyrick Mitchell (Rating: 7.8) (Positional Fit: 99.03)
Malo Gusto (Rating: 7.7) (Positional Fit: 55.94)

Goalkeeper:
Emiliano Martínez (Rating: 8.2)
```

And here is SofaScore's team of the week for the same matchday:
![img_1.png](img_1.png)

As you can see, SofaScore's version is trying to fit as many high rating players into the lineup as possible, leading to the formation having 4 wingers, which is not realistic. While my version has a lower average rating, the formation is much more believable.

One of the issues my model has is that it doesn't have the respective sides of the wide players, so in my team of the week it has 2 left wingers and puts them at right and left wing, so this is something I could change in the future.

Another thing I could possibly change is that I could make an option to do a "team of the season" using players' average ratings over the season and their average positional probabilities.

