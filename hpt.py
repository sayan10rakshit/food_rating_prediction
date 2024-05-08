import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from catboost import CatBoostClassifier

df = pd.read_csv("data/train.csv", encoding="latin1")

# ? Classify int cols, cat columns and target variable
int_cols = df.select_dtypes(include="int64").columns.tolist()
int_cols = [_ for _ in int_cols if _ != "Rating"]
char_cols = df.select_dtypes(include="object").columns.tolist()
target = "Rating"
# print(
#     f"Integer columns: {int_cols} \nCharacter columns: {char_cols} \nTarget: {target}"
# )

df["Recipe_Review"] = df["Recipe_Review"].fillna("")  # ? Fill NaN with empty string

target = df["Rating"]
df = df.drop("Rating", axis=1)  # ? Keep only features

# ? Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.1, random_state=42
)


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    char_cols: list,
    depth: int,
    l2_leaf_reg: float,
    random_strength: float,
    od_wait: int,
    od_type: str,
    leaf_estimation_iterations: int,
    grow_policy: str,
    min_data_in_leaf: int,
    leaf_estimation_method: str,
    num_trees: int,
    return_model: bool = False,
) -> float:
    """
    This function trains a Catboost model on the given data and returns the trained model.

    Args:
        X_train (pd.DataFrame): The training features
        y_train (pd.Series): The training target
        X_test (pd.DataFrame): The test features
        y_test (pd.Series): The test target
        char_cols (list): The list of categorical columns in the dataset
        depth (int): The depth of the trees
        l2_leaf_reg (float): The L2 regularization coefficient
        random_strength (float): The random strength
        od_wait (int): The number of iterations to wait for the metric to improve
        od_type (str): The type of the overfitting detector
        leaf_estimation_iterations (int): The number of iterations to build the leaves
        grow_policy (str): The grow policy
        min_data_in_leaf (int): The minimum number of samples in a leaf
        leaf_estimation_method (str): The method to estimate the leaves
        num_trees (int): The number of trees in the model

    Returns:
        CatBoostClassifier: The trained Catboost model
    """
    # Catboost model
    clf = CatBoostClassifier(
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        od_wait=od_wait,
        od_type=od_type,
        leaf_estimation_iterations=leaf_estimation_iterations,
        grow_policy=grow_policy,
        min_data_in_leaf=min_data_in_leaf,
        leaf_estimation_method=leaf_estimation_method,
        num_trees=num_trees,
        verbose=5000,
        cat_features=char_cols,
        text_features=["RecipeName", "Recipe_Review", "UserName"],
        task_type="GPU",  # ! Uncomment this line and comment the next line to use GPU
        # task_type="CPU",
        text_processing=["tokenize", "stem", "stopwords"],
        loss_function="MultiClass",
    )
    clf.fit(X_train, y_train)

    # ? Predict the labels of the test set and compute the accuracy score
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if return_model:
        return clf
    else:
        return acc


def objective(trial: optuna.Trial) -> float:
    """
    This function defines the objective of the optimization problem.
    It takes a trial object and returns the value of the objective function
    for the given hyperparameters.


    Args:
        trial (optuna.Trial): A trial object that contains the hyperparameters to be sampled

    Returns:
        float: The function returns the accuracy score of the model with the sampled hyperparameters
    """
    # ? Sample the hyperparameters from the trial object
    depth = trial.suggest_int("depth", 6, 10)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 5)
    random_strength = trial.suggest_float("random_strength", 0, 1)
    od_wait = trial.suggest_int("od_wait", 10, 30)
    od_type = trial.suggest_categorical("od_type", ["IncToDec"])
    leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 10)
    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 3)
    leaf_estimation_method = trial.suggest_categorical(
        "leaf_estimation_method", ["Newton", "Gradient"]
    )
    num_trees = trial.suggest_int("num_trees", 8500, 10000)
    acc = train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        char_cols,
        depth,
        l2_leaf_reg,
        random_strength,
        od_wait,
        od_type,
        leaf_estimation_iterations,
        grow_policy,
        min_data_in_leaf,
        leaf_estimation_method,
        num_trees,
    )

    # ? Return the accuracy score as the objective value
    return acc


# ? Create a study object and optimize the objective function
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=100, reduction_factor=3
    ),
)
# study.optimize(objective, n_trials=50)
study.optimize(objective, n_trials=1)

print("\n" * 5)
print("-" * 50)
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
print("-" * 50)

# params = {
#     "depth": 7,
#     "l2_leaf_reg": 4.617145077036919,
#     "random_strength": 0.8619358874883227,
#     "od_wait": 11,
#     "od_type": "IncToDec",
#     "leaf_estimation_iterations": 10,
#     "grow_policy": "SymmetricTree",
#     "min_data_in_leaf": 3,
#     "leaf_estimation_method": "Gradient",
#     "num_trees": 9759,
# }

# model = train_catboost(
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     char_cols,
#     **params,
#     return_model=True,
# )
