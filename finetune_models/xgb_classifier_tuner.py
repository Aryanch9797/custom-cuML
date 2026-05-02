import cudf.pandas
import cuml.accel
cudf.pandas.install()
cuml.accel.install()
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

def build_tuned_XGBClassifier(x, y, n_trials=50):
    """
    Fully tunes and returns the best XGBoost model using 5-fold CV.
    """
    def objective(trial):
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 5.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100.0, log=True),
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, x, y, cv=cv, scoring='f1_weighted', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="XGB_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest XGBoost Hyperparameters: {best_params}")
    
    best_model = xgb.XGBClassifier(**best_params, tree_method='hist', device='cuda', random_state=42)
    return best_model