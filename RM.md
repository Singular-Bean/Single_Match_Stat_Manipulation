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
