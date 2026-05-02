cudf.pandas.install()
cuml.accel.install()
import cuml.accel
import cudf.pandas
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

def build_tuned_KNeighborsClassifier(x, y, n_trials=50):
    """
    Fully tunes and returns the best KNN Classifier using 5-fold CV.
    Safe for cuml.accel GPU fallback limitations.
    """
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            # Only GPU-accelerated metrics
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])
        }
        
        if params['metric'] == 'minkowski':
            params['p'] = trial.suggest_int('p', 3, 5)

        model = KNeighborsClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, x, y, cv=cv, scoring='f1_weighted', n_jobs=1).mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="KNN_Classifier_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nBest KNN Classifier Hyperparameters: {best_params}")
    
    best_model = KNeighborsClassifier(**best_params)
    return best_model