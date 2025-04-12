from src.features.cleaner import clean_data
from src.features.engineering import engineer_features
from src.core.model import load_model
from src.inference.predict import make_predictions

def run_prediction_pipeline(df):
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    model = load_model()  # loads pre-trained model from /models
    predictions = make_predictions(model, df_feat)

    df_clean['churn_prediction'] = predictions
    return df_clean[['customer_id', 'churn_prediction']], df_clean
