# src/inference/predict.py

from src.features.cleaner import DataCleaner
from src.features.engineering import FeatureEngineer
from src.features.feature_selector import FeatureSelector
from src.core.model import ModelHandler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class PredictionEngine:
    def __init__(self, model_path='models/churn_model.pkl', model_type="Automatic (best accuracy)"):
        self.model_handler = ModelHandler(model_path, model_type)
        self.feature_engineer = None
        self.churn_column = None
        self.selected_features = None

    def preprocess(self, df, target_column, feature_columns):
        cleaner = DataCleaner(df)
        df_cleaned = cleaner.clean()
        df_features = df_cleaned.drop(columns=[target_column]) if not feature_columns else df_cleaned[feature_columns]

        self.feature_engineer = FeatureEngineer(target_column=target_column)
        self.feature_engineer.fit(df_features)
        X = self.feature_engineer.transform(df_features)

        return df_cleaned, X

    def apply_feature_selection(self, X, y, methods):
        if not methods:
            return X

        for method in methods:
            selector = FeatureSelector(method=method)
            X = selector.select_features(X, y)

        return X

    def run(self, df, target_column, feature_columns, train_if_no_model=True, feature_selection_method=None):
        df_cleaned, X = self.preprocess(df, target_column, feature_columns)
        y = df_cleaned[target_column]
        self.churn_column = target_column

        if feature_selection_method:
            X = self.apply_feature_selection(X, y, feature_selection_method)
            self.selected_features = list(X.columns)
        else:
            self.selected_features = list(X.columns)

        if train_if_no_model:
            model_name,model_report=self.model_handler.train(X, y)
            self.model_handler.save()
        else:
            self.model_handler.load()
        predictions = self.model_handler.predict(X)
        enriched_df = df_cleaned.copy()

        columns_to_keep = set(self.selected_features + [target_column])
        columns_available = set(enriched_df.columns)

        # Keep only intersecting columns to avoid index errors
        final_columns = list(columns_to_keep & columns_available)
        enriched_df = enriched_df[final_columns].copy()

        # Add predictions
        enriched_df['prediction'] = predictions
        return enriched_df[['prediction']], enriched_df,model_report

    def evaluate_on_test_data(self, test_df, target_column, feature_columns):
        """
        Evaluate the trained model on separate test data
        """
        if self.model_handler.model is None:
            raise ValueError("Model must be trained first before evaluating on test data")
        
        # Preprocess test data using the same feature engineer
        cleaner = DataCleaner(test_df)
        test_df_cleaned = cleaner.clean()
        test_df_features = test_df_cleaned.drop(columns=[target_column]) if not feature_columns else test_df_cleaned[feature_columns]
        
        # Transform test data using the fitted feature engineer
        X_test = self.feature_engineer.transform(test_df_features)
        y_test = test_df_cleaned[target_column]
        
        # Apply same feature selection if it was used during training
        if self.selected_features:
            X_test = X_test[self.selected_features]
        
        # Make predictions
        test_predictions = self.model_handler.predict(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions, average="binary")
        
        # Calculate ROC AUC if model supports predict_proba
        try:
            test_proba = self.model_handler.model.predict_proba(X_test)
            test_auc = roc_auc_score(y_test, test_proba[:, 1])
        except:
            test_auc = float("nan")
        
        # Create test model report
        test_model_report = {
            "Model": f"{self.model_handler.selected_model_name} (Test)",
            "Accuracy": f'{test_accuracy*100:.3f}',
            "F1 Score": f'{test_f1*100:.3f}',
            "ROC AUC": f'{test_auc*100:.3f}'
        }
        
        # Create enriched test dataframe
        test_enriched_df = test_df_cleaned.copy()
        
        # Keep only the columns that exist in both datasets
        columns_to_keep = set(self.selected_features + [target_column])
        columns_available = set(test_enriched_df.columns)
        final_columns = list(columns_to_keep & columns_available)
        test_enriched_df = test_enriched_df[final_columns].copy()
        
        # Add predictions
        test_enriched_df['prediction'] = test_predictions
        
        return test_enriched_df[['prediction']], test_enriched_df, test_model_report

