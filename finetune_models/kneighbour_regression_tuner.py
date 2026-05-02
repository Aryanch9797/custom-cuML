import cudf.pandas
import cuml.accel
cudf.pandas.install()
cuml.accel.install()
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
import optuna

def build_tuned_KNeighborsRegressor(x, y, n_trials=50):
    """
    Fully tunes and returns the best KNN Regressor using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])
        }
        
        if params['metric'] == 'minkowski':
            params['p'] = trial.suggest_int('p', 3, 5)

        model = KNeighborsRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        score = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='minimize', study_name="KNN_Regressor_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest KNN Regressor Hyperparameters: {best_params}")
    
    best_model = KNeighborsRegressor(**best_params)
    return best_model