# src/core/loader.py

import pandas as pd

class FileLoader:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.filename = uploaded_file.name.lower()

    def load(self):
        try:
            if self.filename.endswith('.csv'):
                return pd.read_csv(self.uploaded_file)
            elif self.filename.endswith(('.xlsx', '.xls')):
                return pd.read_excel(self.uploaded_file)
            elif self.filename.endswith('.json'):
                return pd.read_json(self.uploaded_file)
            else:
                raise ValueError("Unsupported file type.")
        except Exception as e:
            raise RuntimeError(f"Error loading file: {e}")
