# src/core/logger.py

import pandas as pd
import os
from datetime import datetime

class AppLogger:
    def __init__(self, log_file='logs/prediction_log.csv'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_prediction(self, filename, num_rows, num_predictions, model_version="v1"):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "filename": filename,
            "rows_uploaded": num_rows,
            "predictions_made": num_predictions,
            "model_version": model_version
        }

        df = pd.DataFrame([log_entry])
        if os.path.exists(self.log_file):
            df.to_csv(self.log_file, mode='a', index=False, header=False)
        else:
            df.to_csv(self.log_file, mode='w', index=False, header=True)
