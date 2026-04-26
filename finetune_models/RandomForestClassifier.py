import pandas as pd
import cuml.accel
cuml.accel.install()
import cudf.pandas
cudf.pandas.install()
from cuml.accel import is_proxy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import optuna


def RandomForestClassifier(x, y, n_trials):
    # Fully tune and return the best model while using 5-fold cross validation
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators',10, 2000)
        max_depth = trial.suggest_int('max_depth', 2, 50) or None
        min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
        score = cross_val_score(model, x, y, cv=5).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params)
    print("Best Hyperparameters:", best_params)
    return best_model
