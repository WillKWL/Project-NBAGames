> (Updated version of [Project-NBASeason](https://github.com/WillKWL/Project-NBASeason))
# Binary classification - Can you predict the outcome of an NBA playoff game 4 months in advance?
- Achieved 56% gain in first 4 deciles after identifying 38 out of 194 features to predict probability of winning an NBA playoff game
- Designed a reusable ML pipeline to prepare dataset, finetune model on train set and evaluate performance on test set
- Common features to predict playoff game outcome shared across top models <img src="../master/data/image/2022-10-02-18-32-13.png">
 
# Problem in answering the original question: 
- Can you tell which team will win the championship before All-Star Game starts?

- **Dataset is too small to predict the outcome**
- There are only 30 NBA teams with historical data starting from 1996-97 season
- That is roughly 500 rows for train set and 200 rows for test set
- You see all kinds of performance issue with this small dataset
  - Bumpy precision-recall curve
  - Overly optimistic AUROC
  - Extremely high variance in model performance
  - <img src="../master/data/image/2022-09-18-16-09-05.png">
  - Even relaxing the target variable from predicting champion to predicting top 4 in playoffs didn't help
    - <img src="../master/data/image/2022-09-28-17-24-03.png">


# Modified business and analytical problem: can you tell which team will win an NBA playoff game before All-Star Game starts?

- Business question
  - By reframing the business question to predicting the outcome of a game instead of NBA champion, we now have more data to use
  - We now have 5000 rows to work with
    - There are roughly 100 playoff games per season
    - 1 row for Win and 1 row for Loss for each game, thus bypassing the need to deal with class imbalance
  - If we can accurately predict the outcome of a game, we can use the result to predict the NBA champion / final 4 teams / final 8 teams etc.
  - Timeline of an NBA season
    - <img src="../master/data/image/2022-09-11-14-37-33.png">

- Analytical problem
  - To predict the probability of a team winning the other team in a playoff game given their respective performance in the regular season before All-Star Game
  - This is a supervised classification problem which can be trained offline

# How will my solution be used
- see [Project-NBASeason #Solution](https://github.com/WillKWL/Project-NBASeason#how-will-my-solution-be-used)

# Background
  
- see [Project-NBASeason #Background](https://github.com/WillKWL/Project-NBASeason#background)


# How should performance be measured
- Model performance = AUPRC (Area under Precision-Recall curve)
  - sklearn implementation: [AP (Average Precision)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
  - alternative: AUROC
- To align with business objective: 
  - Emphasis on precision instead of recall
  - As a team manager / fan, it would be more useful to err on the side of caution and be confident that my prediction is correct, rather than to capture all the winning teams
  - At least 80% precision should be achieved
- Model performance on test set [(more details)](https://github.com/WillKWL/Project-NBAGames/tree/master/source#step-3-evaluation-on-test-set)
  - Precision-Recall Curve <img src="../master/data/image/2022-10-02-18-28-25.png">
  - Average Precision (Area under PR Curve) <img src="../master/data/image/2022-10-02-18-28-38.png"> <img src="../master/data/image/2022-10-02-18-29-58.png">
  - AUROC <img src="../master/data/image/2022-10-03-13-37-15.png">
  - Lift and gain chart <img src="../master/data/image/2022-10-02-18-30-21.png">

# Source of data
- NBA team statistics in regular season and playoffs from [NBA.com/stats](https://www.nba.com/stats/teams/traditional/?sort=W_PCT&dir=-1&Season=2021-22&SeasonType=Regular%20Season&SeasonSegment=Pre%20All-Star) using [nba_api from github](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashteamstats.md)
  - <img src="../master/data/image/2022-09-12-14-53-32.png">
  - Unit for each row = an NBA playoff game
  - X = a team's and its opponent's performance in regular season before All-Star Game <img src="../master/data/image/2022-10-02-18-13-07.png">
  - Y = 0 or 1 with 1 for winning that playoff game
## Type of statistics available
- Base / Advanced / Misc / Four Factors / Scoring / Opponent / Usage / Defense
- Each measures a different aspect of a team's or its opponent's performance in the regular season

## Assumptions 
- setting the scope of data to pre All-Star Game is a good cutoff as mid-season performance
- <img src="../master/data/image/2022-09-12-15-06-12.png">
- Some seasons have abnormally high and low number of games played before All-Star Game, e.g. 1997-98 (82 games) and 1998-99 (4 games) due to NBA lockout
- Removed those abnormal seasons such that all the rows represent a good estimate of mid-season cutoff

# Workflow (more details in /source folder)
1) Train-Test split
2) Exploratory data analysis
3) Data preparation pipeline
  - feature scaling / engineering / selection
  - <img src="../master/data/image/2022-10-03-13-38-24.png">
4) Shortlist potential models
5) Hyperparameter tuning with cross-validation
6) Performance evaluation on test set

# Findings: business objective vs data available
- Most important question
  > - Which team can win the championship?
  - Data you need is each team's performance in a season
  - However, there are only 30 teams for 20 seasons = 600 rows
  - **Dataset is too small**
- How to enlarge our dataset
  > - Reframe the business problem to predicting the outcome of a game instead of NBA champion
  - 200 playoff games for 20 seasons = 4000 rows
  - However, this changes our objective from predicting which team can win **the NBA champion** to which team can win **a playoff game**
  - As we see, a good model for predicting playoff games [might not translate well](https://github.com/WillKWL/Project-NBAGames/tree/master/source#how-well-does-the-prediction-of-playoff-game-outcome-translate-to-predicting-the-ranking-in-playoffs-eg-winning-the-champion) to predicting the NBA champion
- Other findings
  - see [Project-NBASeason #Findings](https://github.com/WillKWL/Project-NBASeason#findings)

# Interesting directions to dig deeper
- Instead of framing as binary classification problem,
- Can we view it as a multi-label classification problem?
  - predicted outcome = rank of a team in playoffs
  - teams may be interested in improving their rank and increasing their chance of winning a championship over time when they don't see it possible to win it right now
- Possible solution
  - e.g. ordinal regression using neural network with multi-label binary classifier output layer + rank conformity check using modified cost function mentioned by [Sebastian Raschka](https://open.spotify.com/episode/772tgKNhb2WCdcccN8IHhm?si=Kn3TcF_zRCS0gBUjJBskNA&utm_source=copy-link&t=970)
