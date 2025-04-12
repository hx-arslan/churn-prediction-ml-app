# src/inference/predict.py

from src.features.cleaner import DataCleaner
from src.features.engineering import FeatureEngineer
from src.core.model import ModelHandler

class PredictionEngine:
    def __init__(self, model_path='models/churn_model.pkl'):
        self.model_handler = ModelHandler(model_path)
        self.feature_engineer = None
        self.churn_column = None

    def run(self, df, train_if_no_model=True):
        # Clean data
        cleaner = DataCleaner(df)
        df_cleaned, churn_col = cleaner.clean()
        self.churn_column = churn_col

        # Prepare feature engineer
        self.feature_engineer = FeatureEngineer(target_column=churn_col)

        # If training is requested or model doesn't exist
        if train_if_no_model:
            if not churn_col:
                raise ValueError("Churn column not found — cannot train model.")
            y = df_cleaned[churn_col]

            # Fit + transform features
            self.feature_engineer.fit(df_cleaned)
            X = self.feature_engineer.transform(df_cleaned)

            # Train and save model
            X_train, X_test, y_train, y_test=self.model_handler.split_data(X,y)
            self.model_handler.train(X_train, y_train)
            self.model_handler.save()
        else:
            # Load model only (user unchecked "retrain")
            self.model_handler.load()
            self.feature_engineer.fit(df_cleaned)  
            X = self.feature_engineer.transform(df_cleaned)

        # Predict
        predictions = self.model_handler.predict(X)
        df_cleaned['churn_prediction'] = predictions

        return df_cleaned[['churn_prediction']], df_cleaned
