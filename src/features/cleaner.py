# src/features/cleaner.py

import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.churn_column = None

    def detect_churn_column(self):
        common_names = ['churn', 'is_churn', 'target', 'churned', 'label']
        for col in self.df.columns:
            if col.lower() in common_names:
                self.churn_column = col
                break
        return self.churn_column

    def clean(self):
        self.df.dropna(axis=1, how='all', inplace=True)

        # Drop duplicate rows
        self.df.drop_duplicates(inplace=True)

        # Forward/backward fill for nulls (can be customized)
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)

        # Detect churn column (if any)
        self.detect_churn_column()

        return self.df, self.churn_column
