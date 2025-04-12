# src/core/model.py

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ModelHandler:
    def __init__(self, model_path='models/churn_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.selected_model_name = None
        
    def split_data(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        best_model = None
        best_accuracy = 0.0
        best_model_name = None

        for name, model in models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            print(f"🔍 {name} Accuracy: {acc:.7f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_name = name

        self.model = best_model
        self.selected_model_name = best_model_name
        print(f"✅ Best model selected: {best_model_name} with accuracy {best_accuracy:.4f}")

        return self.model

    def save(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Trained model not found. Please train it first.")
        self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not loaded or trained.")
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)
