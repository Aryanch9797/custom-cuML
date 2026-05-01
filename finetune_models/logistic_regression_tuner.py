import pandas as pd
import cuml.accel
cuml.accel.install()
import cudf.pandas
cudf.pandas.install()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

def build_tuned_LogisticRegression(x, y, n_trials=50):
    """
    Fully tunes and returns the best Logistic Regression using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-5, 100.0, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'newton-cg']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        }

        model = LogisticRegression(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, x, y, cv=cv, scoring='f1_weighted', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="LogReg_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest Logistic Regression Hyperparameters: {best_params}")
    
    best_model = LogisticRegression(**best_params)
    return best_model