import cudf.pandas
import cuml.accel
cudf.pandas.install()
cuml.accel.install()
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
import gc

def build_tuned_RandomForestClassifier(x, y, n_trials=50):
    """
    Fully tunes and returns the best RandomForest model using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40, 50]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }

        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        score = cross_val_score(model, x, y, cv=cv, scoring='f1_weighted', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="RF_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest RandomForest Hyperparameters: {best_params}")
    
    # Train and return the final model on the best parameters
    best_model = RandomForestClassifier(**best_params, random_state=42)
    return best_model