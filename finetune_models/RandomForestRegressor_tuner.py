import pandas as pd
import cuml.accel
cuml.accel.install()
import cudf.pandas
cudf.pandas.install()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import optuna

def build_tuned_RandomForestRegressor(x, y, n_trials=50):
    """
    Fully tunes and returns the best RandomForestRegressor using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40, 50]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'poisson']),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }

        model = RandomForestRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # neg_mean_squared_error returns negative values, so we multiply by -1 to get positive MSE
        score = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean()
        return score

    # We want the lowest MSE possible
    study = optuna.create_study(direction='minimize', study_name="RF_Regressor_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest RandomForestRegressor Hyperparameters: {best_params}")
    
    best_model = RandomForestRegressor(**best_params, random_state=42)
    return best_model