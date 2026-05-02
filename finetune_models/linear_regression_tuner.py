cudf.pandas.install()
cuml.accel.install()
import cuml.accel
import cudf.pandas
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import optuna

def build_tuned_LinearRegression(x, y, n_trials=5):
    """
    Fully tunes and returns the best Linear Regression using 5-fold CV.
    (Note: Linear Regression has very limited tuning space).
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
        }

        model = LinearRegression(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        score = -cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean()
        return score

    # Very small n_trials needed since there are only 2 possible combinations
    study = optuna.create_study(direction='minimize', study_name="LinearRegression_Tuning")
    study.optimize(objective, n_trials=min(n_trials, 5)) 
    
    best_params = study.best_params
    print(f"\nBest Linear Regression Hyperparameters: {best_params}")
    
    best_model = LinearRegression(**best_params)
    return best_model