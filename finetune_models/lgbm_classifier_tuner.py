import pandas as pd
import cudf.pandas
cudf.pandas.install()

import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

def build_tuned_LGBMClassifier(x, y, n_trials=50):
    """
    Fully tunes and returns the best LightGBM model using 5-fold CV.
    """
    def objective(trial):
        params = {
            'device': 'gpu',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.2),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100.0, log=True),
            'verbose': -1,
            'random_state': 42
        }

        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, x, y, cv=cv, scoring='f1_weighted', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="LGBM_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest LightGBM Hyperparameters: {best_params}")
    
    best_model = lgb.LGBMClassifier(**best_params, device='gpu', verbose=-1, random_state=42)
    return best_model