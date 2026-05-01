import pandas as pd
import cuml.accel
cuml.accel.install()
import cudf.pandas
cudf.pandas.install()

from cuml.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

def build_tuned_SVC(x, y, n_trials=50):
    """
    Fully tunes and returns the best SVC model using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1000.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),
            'probability': True  # Required for your predict_proba usage
        }
        
        if params['kernel'] == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 6)
            params['coef0'] = trial.suggest_float('coef0', -10.0, 10.0)
        elif params['kernel'] == 'sigmoid':
            params['coef0'] = trial.suggest_float('coef0', -10.0, 10.0)

        model = SVC(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, x, y, cv=cv, scoring='f1_weighted', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="SVC_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest SVC Hyperparameters: {best_params}")
    
    best_model = SVC(**best_params, probability=True)
    return best_model