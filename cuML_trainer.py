import pandas as pd
import cuml.accel
cuml.accel.install()
import cudf.pandas
cudf.pandas.install()
from cuml.accel import is_proxy
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
 

class CuMLTrainer:
    def __init__(self,x,y,x_test=None):
        self.x = x
        self.y = y
        self.x_test = x_test
        
    def RandomForestClassifier(self, fine_tune=False, n_trials=25):
        if fine_tune:
            from finetune_models.RandomForestClassifier import RandomForestClassifier
            model = RandomForestClassifier(None, self.x, self.y, n_trials)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()