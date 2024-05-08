# Recipe for Rating: Predict Food Ratings using ML

 Predicting food ratings from user's recipe reviews.


## Competition Description

- Visit the competition page ➡️ [competition page](https://www.kaggle.com/competitions/recipe-for-rating-predict-food-ratings-using-ml)

- Visit Leaderboard ➡️ [leaderboard](https://www.kaggle.com/competitions/recipe-for-rating-predict-food-ratings-using-ml/leaderboard)

**Rank - 10
Team Name - Sayan Rakshit**

The competition is to predict the rating of a recipe given the user's review. The dataset contains the following columns:

- **RecipeNumber**: Placement of the recipe on the top 100 recipes list
- **RecipeCode**: Unique ID of the recipe used by the site
- **RecipeName**: Name of the recipe the comment was posted on
- **CommentID**: Unique ID of the comment
- **UserID**: Unique ID of the user who left the comment
- **UserName**: Name of the user
- **UserReputation**: Internal score of the site, roughly quantifying the past behavior of the user
- **CreationTimestamp**: Time at which the comment was posted as a Unix timestamp
- **ReplyCount**: Number of replies to the comment
- **ThumbsUpCount**: Number of up-votes the comment has received
- **ThumbsDownCount**: Number of down-votes the comment has received
- **Rating**: The score on a 1 to 5 scale that the user gave to the recipe. A score of 0 means that no score was given (Target Variable)
- **BestScore**: Score of the comment, likely used by the site to help determine the order comments appear in
- **Recipe_Review**: Text content of the comment

## Dependencies

- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Optuna
- Catboost

> [!NOTE]
> This competition was hosted by MLP Project team of IIT Madras BS Degree [^1].

[^1]: [study.iitm.ac.in/ds](https://study.iitm.ac.in/ds/course_pages/BSCS2008P.html)
