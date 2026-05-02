cudf.pandas.install()
cuml.accel.install()
import cuml.accel
import cudf.pandas
import pandas as pd

from cuml.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
import optuna

def build_tuned_SVR(x, y, n_trials=50):
    """
    Fully tunes and returns the best SVR (Support Vector Regressor) model using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1000.0, log=True),
            # Epsilon defines the "tube" within which no penalty is associated in the training loss function
            'epsilon': trial.suggest_float('epsilon', 1e-4, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True)
        }
        
        # Degree and coef0 are only mathematically applied to poly and sigmoid kernels
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 6)
            params['coef0'] = trial.suggest_float('coef0', -10.0, 10.0)
        elif params['kernel'] == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', -10.0, 10.0)

        model = SVR(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        score = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='minimize', study_name="SVR_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest SVR Hyperparameters: {best_params}")
    
    best_model = SVR(**best_params)
    return best_model