import pandas as pd
import cuml.accel
cuml.accel.install()
import cudf.pandas
cudf.pandas.install()
from cuml.accel import is_proxy
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

 

class CuMLTrainer:
    def __init__(self,x,y,x_test=None):
        self.x = x
        self.y = y
        self.x_test = x_test

    def classification_splits(self, model):
        oof_pred_proba = np.zeros((len(self.x), len(np.unique(self.y))))
        if self.x_test is not None:
            test_pred_proba = np.zeros((len(self.x_test), len(np.unique(self.y))))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.x, self.y)):
            X_train, y_train = self.x.iloc[train_idx], self.y.iloc[train_idx]
            X_val, y_val = self.x.iloc[val_idx], self.y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_val_pred_proba = model.predict_proba(X_val)
            oof_pred_proba[val_idx] = y_val_pred_proba
            if self.x_test is not None:
                test_pred_proba += model.predict_proba(self.x_test) / skf.n_splits

            val_final_preds = np.argmax(y_val_pred_proba, axis=1)
            print(f"Fold {fold + 1}, accuracy: {accuracy_score(y_val, val_final_preds)}, f1: {f1_score(y_val, val_final_preds, average='weighted')}, precision: {precision_score(y_val, val_final_preds, average='weighted')}, recall: {recall_score(y_val, val_final_preds, average='weighted')}")
        
        # Final oof results
        oof_final_preds = np.argmax(oof_pred_proba, axis=1)
        print(f"OOF accuracy: {accuracy_score(self.y, oof_final_preds)}, f1: {f1_score(self.y, oof_final_preds, average='weighted')}, precision: {precision_score(self.y, oof_final_preds, average='weighted')}, recall: {recall_score(self.y, oof_final_preds, average='weighted')}")
        print("Classification Report:\n", classification_report(self.y, oof_final_preds))
        sns.heatmap(confusion_matrix(self.y, oof_final_preds), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        return oof_pred_proba, test_pred_proba
    
    def regression_splits(self, model):
        oof_pred = np.zeros(len(self.x))
        if self.x_test is not None:
            test_pred = np.zeros(len(self.x_test))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.x)):
            X_train, y_train = self.x.iloc[train_idx], self.y.iloc[train_idx]
            X_val, y_val = self.x.iloc[val_idx], self.y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            oof_pred[val_idx] = y_val_pred
            if self.x_test is not None:
                test_pred += model.predict(self.x_test) / kf.n_splits

            print(f"Fold {fold + 1}, MSE: {mean_squared_error(y_val, y_val_pred)}, R2: {r2_score(y_val, y_val_pred)}, MAE: {mean_absolute_error(y_val, y_val_pred)}")
        # Final oof results
        print(f"OOF MSE: {mean_squared_error(self.y, oof_pred)}, R2: {r2_score(self.y, oof_pred)}, MAE: {mean_absolute_error(self.y, oof_pred)}")

        return oof_pred, test_pred

    def train_RandomForestClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.RandomForestClassifier import RandomForestClassifier
            model = RandomForestClassifier(None, self.x, self.y, n_trials)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()

        oof_pred_proba, test_pred_proba = self.classification_splits(model)
        
        return oof_pred_proba, test_pred_proba
    
    def train_RandomForestRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.RandomForestRegressor import RandomForestRegressor
            model = RandomForestRegressor(None, self.x, self.y, n_trials)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        
        oof_pred, test_pred = self.regression_splits(model)

        return oof_pred, test_pred
    
    def train_LinearRegression(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.LinearRegression import LinearRegression
            model = LinearRegression(None, self.x, self.y, n_trials)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        
        oof_pred, test_pred = self.regression_splits(model)

        return oof_pred, test_pred

    
    def train_LogisticRegression(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.LogisticRegression import LogisticRegression
            model = LogisticRegression(None, self.x, self.y, n_trials)
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model)
        return oof_pred_proba, test_pred_proba
    
    def train_ElasticNet(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.ElasticNet import ElasticNet
            model = ElasticNet(None, self.x, self.y, n_trials)
        else:
            from sklearn.linear_model import ElasticNet
            model = ElasticNet()
        
        oof_pred, test_pred = self.regression_splits(model)
        return oof_pred, test_pred

    
    def train_KNeighborsClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.KNeighborsClassifier import KNeighborsClassifier
            model = KNeighborsClassifier(None, self.x, self.y, n_trials)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model)
        return oof_pred_proba, test_pred_proba
    
    def train_KNeighborsRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.KNeighborsRegressor import KNeighborsRegressor
            model = KNeighborsRegressor(None, self.x, self.y, n_trials)
        else:
            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor()
        
        oof_pred, test_pred = self.regression_splits(model)
        return oof_pred, test_pred

    def Baseline_comparison_regressor(self):
        scores = {}
        rf_oof_pred, rf_test_pred = self.train_RandomForestRegressor()
        lr_oof_pred, lr_test_pred = self.train_LinearRegression()
        en_oof_pred, en_test_pred = self.train_ElasticNet()
        knn_reg_oof_pred, knn_reg_test_pred = self.train_KNeighborsRegressor()

        scores['RandomForestRegressor'] = (mean_squared_error(self.y, rf_oof_pred), r2_score(self.y, rf_oof_pred), mean_absolute_error(self.y, rf_oof_pred))
        scores['LinearRegression'] = (mean_squared_error(self.y, lr_oof_pred), r2_score(self.y, lr_oof_pred), mean_absolute_error(self.y, lr_oof_pred))
        scores['ElasticNet'] = (mean_squared_error(self.y, en_oof_pred), r2_score(self.y, en_oof_pred), mean_absolute_error(self.y, en_oof_pred))
        scores['KNeighborsRegressor'] = (mean_squared_error(self.y, knn_reg_oof_pred), r2_score(self.y, knn_reg_oof_pred), mean_absolute_error(self.y, knn_reg_oof_pred))   

        # Comparision plots
        metrics = ['MSE', 'R2', 'MAE']
        for i, metric in enumerate(metrics):
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(scores.keys()), y=[scores[model][i] for model in scores])
            plt.title(f'Model Comparison - {metric}')
            plt.ylabel(metric)
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.show()
        return scores
    
    def Baseline_comparison_classifier(self):
        scores = {}
        rf_oof_pred_proba, rf_test_pred_proba = self.train_RandomForestClassifier()
        lr_oof_pred_proba, lr_test_pred_proba = self.train_LogisticRegression()
        knn_oof_pred_proba, knn_test_pred_proba = self.train_KNeighborsClassifier()

        rf_oof_final_preds = np.argmax(rf_oof_pred_proba, axis=1)
        lr_oof_final_preds = np.argmax(lr_oof_pred_proba, axis=1)
        knn_oof_final_preds = np.argmax(knn_oof_pred_proba, axis=1)

        scores['RandomForestClassifier'] = (accuracy_score(self.y, rf_oof_final_preds), f1_score(self.y, rf_oof_final_preds, average='weighted'), precision_score(self.y, rf_oof_final_preds, average='weighted'), recall_score(self.y, rf_oof_final_preds, average='weighted'))
        scores['LogisticRegression'] = (accuracy_score(self.y, lr_oof_final_preds), f1_score(self.y, lr_oof_final_preds, average='weighted'), precision_score(self.y, lr_oof_final_preds, average='weighted'), recall_score(self.y, lr_oof_final_preds, average='weighted'))
        scores['KNeighborsClassifier'] = (accuracy_score(self.y, knn_oof_final_preds), f1_score(self.y, knn_oof_final_preds, average='weighted'), precision_score(self.y, knn_oof_final_preds, average='weighted'), recall_score(self.y, knn_oof_final_preds, average='weighted'))   

        # Comparision plots
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        for i, metric in enumerate(metrics):
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(scores.keys()), y=[scores[model][i] for model in scores])
            plt.title(f'Model Comparison - {metric}')
            plt.ylabel(metric)
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.show()

        # Print classification reports
        for model in scores:
            print(f"Classification Report for {model}:\n", classification_report(self.y, np.argmax(eval(f"{model.lower()}_oof_pred_proba"), axis=1)))

        # Confusion matrices
        models = ["rf", "lr", "knn"]
        for model in models:
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(self.y, np.argmax(eval(f"{model.lower()}_oof_pred_proba"), axis=1)), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
        return scores



            