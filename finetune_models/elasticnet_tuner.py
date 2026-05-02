cudf.pandas.install()
cuml.accel.install()
import cuml.accel
import cudf.pandas
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
import optuna

def build_tuned_ElasticNet(x, y, n_trials=50):
    """
    Fully tunes and returns the best ElasticNet Regressor using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 100.0, log=True),
            # l1_ratio=1 is pure Lasso, l1_ratio=0 is pure Ridge
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'random_state': 42
        }

        model = ElasticNet(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        score = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='minimize', study_name="ElasticNet_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest ElasticNet Hyperparameters: {best_params}")
    
    best_model = ElasticNet(**best_params, random_state=42)
    return best_model