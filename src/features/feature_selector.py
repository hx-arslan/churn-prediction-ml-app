# src/features/feature_selector.py

import pandas as pd
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, method='RFE', num_features=None, threshold=0.01):
        self.method = method
        self.num_features = num_features
        self.threshold = threshold
        self.selected_features = []

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.method == 'RFE':
            estimator = LogisticRegression(max_iter=1000)
            selector = RFE(estimator, n_features_to_select=self.num_features or int(X.shape[1] / 2))
            selector.fit(X, y)
            mask = selector.support_

        elif self.method == 'Lasso':
            estimator = Lasso(alpha=0.01)
            selector = SelectFromModel(estimator)
            selector.fit(X, y)
            mask = selector.get_support()

        elif self.method == 'VarianceThreshold':
            selector = VarianceThreshold(threshold=self.threshold)
            selector.fit(X)
            mask = selector.get_support()

        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

        self.selected_features = X.columns[mask].tolist()
        return X[self.selected_features]

    def get_selected_features(self):
        return self.selected_features