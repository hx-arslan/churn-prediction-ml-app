# src/inference/predict.py

from src.features.cleaner import DataCleaner
from src.features.engineering import FeatureEngineer
from src.core.model import ModelHandler

class PredictionEngine:
    def __init__(self, model_path='models/churn_model.pkl',model_type="Automatic (best accuracy)"):
        self.model_handler = ModelHandler(model_path,model_type)
        self.feature_engineer = None
        self.churn_column = None

    def run(self, df,target_column,feature_columns, train_if_no_model=True):
        # Clean data
        cleaner = DataCleaner(df)
        df_cleaned = cleaner.clean()
        df_features=df_cleaned.drop(columns=[target_column]) if not feature_columns else df_cleaned[feature_columns]
        self.churn_column = target_column

        # Prepare feature engineer
        self.feature_engineer = FeatureEngineer(target_column=target_column)

        # If training is requested or model doesn't exist
        if train_if_no_model:
            if not target_column:
                raise ValueError("Target column not found — cannot train model.")
            y = df_cleaned[target_column]

            # Fit + transform features
            self.feature_engineer.fit(df_features)
            X = self.feature_engineer.transform(df_features)

            # Train and save model
            self.model_handler.train(X, y)
            self.model_handler.save()
        else:
            # Load model only (user unchecked "retrain")
            self.model_handler.load()
            self.feature_engineer.fit(df_features)  
            X = self.feature_engineer.transform(df_features)

        # Predict
        predictions = self.model_handler.predict(X)
        df_cleaned['prediction'] = predictions

        return df_cleaned[['prediction']], df_cleaned
