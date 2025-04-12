# src/features/engineering.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self, target_column: str = None):
        self.target_column = target_column
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_columns = []

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        if self.target_column and self.target_column in df.columns:
            df.drop(columns=[self.target_column], inplace=True)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        self.scaler.fit(df[numeric_cols])
        self.ohe.fit(df[categorical_cols])

        # Save feature column order
        numeric_names = list(numeric_cols)
        categorical_names = list(self.ohe.get_feature_names_out(categorical_cols))
        self.feature_columns = numeric_names + categorical_names
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform.")

        df = df.copy()
        if self.target_column and self.target_column in df.columns:
            df.drop(columns=[self.target_column], inplace=True)

        # Handle numeric
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_scaled = pd.DataFrame(
            self.scaler.transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )

        # Handle categorical
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_encoded = pd.DataFrame(
            self.ohe.transform(df[categorical_cols]),
            columns=self.ohe.get_feature_names_out(categorical_cols),
            index=df.index
        )

        # Combine
        X = pd.concat([numeric_scaled, categorical_encoded], axis=1)

        # Reindex to match training columns (fill missing with 0)
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        return X
