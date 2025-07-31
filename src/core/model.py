# src/core/model.py

import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

class ModelHandler:
    def __init__(self, model_path='models/churn_model.pkl',model_type="Automatic (best accuracy)"):
        self.model_path = model_path
        self.model_type=model_type
        self.model = None
        self.selected_model_name = None
        
    def split_data(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        # Optimized model configurations for better performance
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            "Support Vector Machine": SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }

        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.model_report = []
        self.best_model_report=None
        
        if self.model_type == "Automatic (best accuracy)":
            best_accuracy = 0.0
            best_model = None
            best_model_name = None

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="binary")
                    
                    # Calculate ROC AUC safely
                    try:
                        if hasattr(model, "predict_proba"):
                            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                        else:
                            auc = float("nan")
                    except:
                        auc = float("nan")

                    self.model_report.append({
                        "Model": name,
                        "Accuracy": acc*100,
                        "F1 Score": f1*100,
                        "ROC AUC": auc*100
                    })

                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = model
                        best_model_name = name
                        
                except Exception as e:
                    print(f"⚠️ Error training {name}: {e}")

            self.model = best_model
            self.selected_model_name = best_model_name

        else:
            model = models.get(self.model_type)
            if model is None:
                raise ValueError(f"Unknown model type: {self.model_type}")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="binary")
                
                # Calculate ROC AUC safely
                try:
                    if hasattr(model, "predict_proba"):
                        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                    else:
                        auc = float("nan")
                except:
                    auc = float("nan")

                self.model_report.append({
                    "Model": self.model_type,
                    "Accuracy": acc*100,
                    "F1 Score": f1*100,
                    "ROC AUC": auc*100
                })

                self.model = model
                self.selected_model_name = self.model_type

            except Exception as e:
                print(f"⚠️ Error training selected model {self.model_type}: {e}")
                raise e

        print(f"\n✅ Selected Model: {self.selected_model_name}")
        for r in self.model_report:
            print(f"{r['Model']:25} | Acc: {r['Accuracy']:.3f} | F1: {r['F1 Score']:.3f} | AUC: {r['ROC AUC']:.3f}")
            if r['Model']==self.selected_model_name:
                self.best_model_report={
                    "Model": r['Model'],
                    "Accuracy": f'{r['Accuracy']:.3f}',
                    "F1 Score": f'{r['F1 Score']:.3f}',
                    "ROC AUC": f'{r['ROC AUC']:.3f}'
                }
        return self.model,self.best_model_report

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
