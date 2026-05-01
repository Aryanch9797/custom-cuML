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
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from cuml.svm import SVC, SVR
 

class CuMLTrainer:
    def __init__(self,x,y,x_test=None):
        self.x = x
        self.y = y
        self.x_test = x_test

    def classification_splits(self, model, model_name="Model"):
        oof_pred_proba = np.zeros((len(self.x), len(np.unique(self.y))))
        if self.x_test is not None:
            test_pred_proba = np.zeros((len(self.x_test), len(np.unique(self.y))))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.x, self.y)):
            if hasattr(self.x, 'iloc'):
                X_train, y_train = self.x.iloc[train_idx], self.y.iloc[train_idx]
                X_val, y_val = self.x.iloc[val_idx], self.y.iloc[val_idx]
            else:
                X_train, y_train = self.x[train_idx], self.y[train_idx]
                X_val, y_val = self.x[val_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            y_val_pred_proba = model.predict_proba(X_val)
            oof_pred_proba[val_idx] = y_val_pred_proba
            if self.x_test is not None:
                test_pred_proba += model.predict_proba(self.x_test) / skf.n_splits

            val_final_preds = np.argmax(y_val_pred_proba, axis=1)
            print(f"Fold {fold + 1}, accuracy: {accuracy_score(y_val, val_final_preds)}, f1: {f1_score(y_val, val_final_preds, average='weighted')}, precision: {precision_score(y_val, val_final_preds, average='weighted')}, recall: {recall_score(y_val, val_final_preds, average='weighted')}")
        
        # Final oof results
        oof_final_preds = np.argmax(oof_pred_proba, axis=1)
        print(f"{model_name} OOF accuracy: {accuracy_score(self.y, oof_final_preds)}, f1: {f1_score(self.y, oof_final_preds, average='weighted')}, precision: {precision_score(self.y, oof_final_preds, average='weighted')}, recall: {recall_score(self.y, oof_final_preds, average='weighted')}")
        print(f" {model_name} Classification Report:\n", classification_report(self.y, oof_final_preds))
        sns.heatmap(confusion_matrix(self.y, oof_final_preds), annot=True, fmt='d', cmap='Blues')
        plt.title(f' {model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        return oof_pred_proba, test_pred_proba if self.x_test is not None else None
    
    def regression_splits(self, model, model_name="Model"):
        oof_pred = np.zeros(len(self.x))
        if self.x_test is not None:
            test_pred = np.zeros(len(self.x_test))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.x)):
            if hasattr(self.x, 'iloc'):
                X_train, y_train = self.x.iloc[train_idx], self.y.iloc[train_idx]
                X_val, y_val = self.x.iloc[val_idx], self.y.iloc[val_idx]
            else:
                X_train, y_train = self.x[train_idx], self.y[train_idx]
                X_val, y_val = self.x[val_idx], self.y[val_idx]
            
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            oof_pred[val_idx] = y_val_pred
            if self.x_test is not None:
                test_pred += model.predict(self.x_test) / kf.n_splits

            print(f"Fold {fold + 1}, MSE: {mean_squared_error(y_val, y_val_pred)}, R2: {r2_score(y_val, y_val_pred)}, MAE: {mean_absolute_error(y_val, y_val_pred)}")
        # Final oof results
        print(f"{model_name} OOF MSE: {mean_squared_error(self.y, oof_pred)}, R2: {r2_score(self.y, oof_pred)}, MAE: {mean_absolute_error(self.y, oof_pred)}")

        return oof_pred, test_pred if self.x_test is not None else None

    def train_RandomForestClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.RandomForestClassifier_tuner import build_tuned_RandomForestClassifier
            model = build_tuned_RandomForestClassifier(self.x, self.y, n_trials)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()

        oof_pred_proba, test_pred_proba = self.classification_splits(model, "RandomForestClassifier")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'rf_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'rf_test_pred_finetune{fine_tune}.csv')
        
        return oof_pred_proba, test_pred_proba
    
    def train_RandomForestRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.RandomForestRegressor_tuner import build_tuned_RandomForestRegressor
            model = build_tuned_RandomForestRegressor(self.x, self.y, n_trials)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        
        oof_pred, test_pred = self.regression_splits(model, "RandomForestRegressor")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'rf_reg_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'rf_reg_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred
    
    def train_LinearRegression(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.linear_regression_tuner import build_tuned_LinearRegression
            model = build_tuned_LinearRegression(self.x, self.y, n_trials)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        
        oof_pred, test_pred = self.regression_splits(model, "LinearRegression")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'lr_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'lr_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred

    
    def train_LogisticRegression(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.logistic_regression_tuner import build_tuned_LogisticRegression
            model = build_tuned_LogisticRegression(self.x, self.y, n_trials)
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model, "LogisticRegression")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'lr_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'lr_test_pred_finetune{fine_tune}.csv')

        return oof_pred_proba, test_pred_proba
    
    def train_ElasticNet(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.elasticnet_tuner import build_tuned_ElasticNet
            model = build_tuned_ElasticNet(self.x, self.y, n_trials)
        else:
            from sklearn.linear_model import ElasticNet
            model = ElasticNet()
        
        oof_pred, test_pred = self.regression_splits(model, "ElasticNet")
        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'en_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'en_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred

    
    def train_KNeighborsClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.kneighbour_classifier_tuner import build_tuned_KNeighborsClassifier
            model = build_tuned_KNeighborsClassifier(self.x, self.y, n_trials)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model, "KNeighborsClassifier")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'knc_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'knc_test_pred_finetune{fine_tune}.csv')

        return oof_pred_proba, test_pred_proba
    
    
    def train_KNeighborsRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.kneighbour_regression_tuner import build_tuned_KNeighborsRegressor
            model = build_tuned_KNeighborsRegressor(self.x, self.y, n_trials)
        else:
            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor()
        
        oof_pred, test_pred = self.regression_splits(model, "KNeighborsRegressor")
        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'knn_reg_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'knn_reg_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred
    
    def train_XGBClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.xgb_classifier_tuner import build_tuned_XGBClassifier
            model = build_tuned_XGBClassifier(self.x, self.y, n_trials)
        else:
            model = xgb.XGBClassifier(
                tree_method='hist', 
                device='cuda', 
                random_state=42
            )
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model, "XGBClassifier")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'xgb_clf_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'xgb_clf_test_pred_finetune{fine_tune}.csv')

        return oof_pred_proba, test_pred_proba

    def train_XGBRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.xgb_regression_tuner import build_tuned_XGBRegressor
            model = build_tuned_XGBRegressor(self.x, self.y, n_trials)
        else:
            model = xgb.XGBRegressor(
                tree_method='hist', 
                device='cuda', 
                random_state=42
            )
        
        oof_pred, test_pred = self.regression_splits(model, "XGBRegressor")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'xgb_reg_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'xgb_reg_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred
    
    def train_LGBMClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.lgbm_classifier_tuner import build_tuned_LGBMClassifier
            model = build_tuned_LGBMClassifier(self.x, self.y, n_trials)
        else:
            model = lgb.LGBMClassifier(
                device='gpu', 
                random_state=42,
                verbose=-1 
            )
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model, "LGBMClassifier")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'lgb_clf_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'lgb_clf_test_pred_finetune{fine_tune}.csv')

        return oof_pred_proba, test_pred_proba

    def train_LGBMRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.lgbm_regression_tuner import build_tuned_LGBMRegressor
            model = build_tuned_LGBMRegressor(self.x, self.y, n_trials)
        else:
            model = lgb.LGBMRegressor(
                device='gpu', 
                random_state=42,
                verbose=-1
            )
        
        oof_pred, test_pred = self.regression_splits(model, "LGBMRegressor")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'lgb_reg_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'lgb_reg_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred
    
    def train_CatBoostClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.cat_classifier_tuner import build_tuned_CatBoostClassifier
            model = build_tuned_CatBoostClassifier(self.x, self.y, n_trials)
        else:
            model = cb.CatBoostClassifier(
                task_type='GPU', 
                random_state=42,
                verbose=0
            )
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model, "CatBoostClassifier")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'catboost_clf_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'catboost_clf_test_pred_finetune{fine_tune}.csv')

        return oof_pred_proba, test_pred_proba

    def train_CatBoostRegressor(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.cat_regression_tuner import build_tuned_CatBoostRegressor
            model = build_tuned_CatBoostRegressor(self.x, self.y, n_trials)
        else:
            model = cb.CatBoostRegressor(
                task_type='GPU', 
                random_state=42,
                verbose=0
            )
        
        oof_pred, test_pred = self.regression_splits(model, "CatBoostRegressor")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'catboost_reg_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'catboost_reg_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred
    
    def train_SVC(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.svc_tuner import build_tuned_SVC
            model = build_tuned_SVC(self.x, self.y, n_trials)
        else:
            model = SVC(probability=True) 
        
        oof_pred_proba, test_pred_proba = self.classification_splits(model, "cuML_SVC")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred_proba).to_csv(f'svc_oof_pred_finetune{fine_tune}.csv')
        if test_pred_proba is not None:
            pd.DataFrame(test_pred_proba).to_csv(f'svc_test_pred_finetune{fine_tune}.csv')

        return oof_pred_proba, test_pred_proba

    def train_SVR(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.svr_tuner import build_tuned_SVR
            model = build_tuned_SVR(self.x, self.y, n_trials)
        else:
            model = SVR()
        
        oof_pred, test_pred = self.regression_splits(model, "cuML_SVR")

        # saving oof prediction and test prediction as csv
        pd.DataFrame(oof_pred).to_csv(f'svr_oof_pred_finetune{fine_tune}.csv')
        if test_pred is not None:
            pd.DataFrame(test_pred).to_csv(f'svr_test_pred_finetune{fine_tune}.csv')

        return oof_pred, test_pred
    

    def Baseline_comparison_regressor(self, fine_tune=False, n_trials=25):
        scores = {}
        rf_oof_pred, rf_test_pred = self.train_RandomForestRegressor(fine_tune, n_trials)
        lr_oof_pred, lr_test_pred = self.train_LinearRegression(fine_tune, n_trials)
        en_oof_pred, en_test_pred = self.train_ElasticNet(fine_tune, n_trials)
        knn_reg_oof_pred, knn_reg_test_pred = self.train_KNeighborsRegressor(fine_tune, n_trials)
        xgb_reg_oof_pred, xgb_reg_test_pred = self.train_XGBRegressor(fine_tune, n_trials)

        scores['RandomForestRegressor'] = (mean_squared_error(self.y, rf_oof_pred), r2_score(self.y, rf_oof_pred), mean_absolute_error(self.y, rf_oof_pred))
        scores['LinearRegression'] = (mean_squared_error(self.y, lr_oof_pred), r2_score(self.y, lr_oof_pred), mean_absolute_error(self.y, lr_oof_pred))
        scores['ElasticNet'] = (mean_squared_error(self.y, en_oof_pred), r2_score(self.y, en_oof_pred), mean_absolute_error(self.y, en_oof_pred))
        scores['KNeighborsRegressor'] = (mean_squared_error(self.y, knn_reg_oof_pred), r2_score(self.y, knn_reg_oof_pred), mean_absolute_error(self.y, knn_reg_oof_pred))   
        scores['XGBRegressor'] = (mean_squared_error(self.y, xgb_reg_oof_pred), r2_score(self.y, xgb_reg_oof_pred), mean_absolute_error(self.y, xgb_reg_oof_pred))

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
    
    def Baseline_comparison_classifier(self, fine_tune=False, n_trials=25):
        scores = {}
        oof_preds = {}

        rf_oof_pred, rf_test_pred = self.train_RandomForestClassifier(fine_tune, n_trials)
        lr_oof_pred, lr_test_pred = self.train_LogisticRegression(fine_tune, n_trials)
        knn_oof_pred, knn_test_pred = self.train_KNeighborsClassifier(fine_tune, n_trials)
        xgb_oof_pred, xgb_test_pred = self.train_XGBClassifier(fine_tune, n_trials)

        oof_preds['rf'] = rf_oof_pred
        oof_preds['lr'] = lr_oof_pred
        oof_preds['knn'] = knn_oof_pred
        oof_preds['xgb'] = xgb_oof_pred

        rf_oof_final_preds = np.argmax(oof_preds['rf'][0], axis=1)
        lr_oof_final_preds = np.argmax(oof_preds['lr'][0], axis=1)
        knn_oof_final_preds = np.argmax(oof_preds['knn'][0], axis=1)
        xgb_oof_final_preds = np.argmax(oof_preds['xgb'][0], axis=1)

        scores['RandomForestClassifier'] = (accuracy_score(self.y, rf_oof_final_preds), f1_score(self.y, rf_oof_final_preds, average='weighted'), precision_score(self.y, rf_oof_final_preds, average='weighted'), recall_score(self.y, rf_oof_final_preds, average='weighted'))
        scores['LogisticRegression'] = (accuracy_score(self.y, lr_oof_final_preds), f1_score(self.y, lr_oof_final_preds, average='weighted'), precision_score(self.y, lr_oof_final_preds, average='weighted'), recall_score(self.y, lr_oof_final_preds, average='weighted'))
        scores['KNeighborsClassifier'] = (accuracy_score(self.y, knn_oof_final_preds), f1_score(self.y, knn_oof_final_preds, average='weighted'), precision_score(self.y, knn_oof_final_preds, average='weighted'), recall_score(self.y, knn_oof_final_preds, average='weighted'))
        scores['XGBClassifier'] = (accuracy_score(self.y, xgb_oof_final_preds), f1_score(self.y, xgb_oof_final_preds, average='weighted'), precision_score(self.y, xgb_oof_final_preds, average='weighted'), recall_score(self.y, xgb_oof_final_preds, average='weighted'))

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
            print(f"Classification Report for {model}:\n", classification_report(self.y, np.argmax(oof_preds[model][0], axis=1)))

        # Confusion matrices
        models = ["rf", "lr", "knn", "xgb"]
        for model in models:
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(self.y, np.argmax(oof_preds[model][0], axis=1)), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        return scores



            