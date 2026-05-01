import pandas as pd
import cudf.pandas
cudf.pandas.install()

import catboost as cb
from sklearn.model_selection import cross_val_score, KFold
import optuna

def build_tuned_CatBoostRegressor(x, y, n_trials=50):
    """
    Fully tunes and returns the best CatBoost Regressor using 5-fold CV.
    """
    def objective(trial):
        params = {
            'task_type': 'GPU',
            'iterations': trial.suggest_int('iterations', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 100.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'verbose': 0,
            'random_state': 42
        }

        model = cb.CatBoostRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        score = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='minimize', study_name="CatBoost_Regressor_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest CatBoost Regressor Hyperparameters: {best_params}")
    
    best_model = cb.CatBoostRegressor(**best_params, task_type='GPU', verbose=0, random_state=42)
    return best_model