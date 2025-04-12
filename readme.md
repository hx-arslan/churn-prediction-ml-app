# 📉 Churn Prediction App

A dynamic, user-friendly Streamlit application for predicting customer churn using machine learning. Designed to support non-technical users with file uploads, real-time predictions, retraining, and insightful visualizations.

---

## 🚀 Features

- 📁 **File Upload**: Upload CSV, Excel, or JSON files directly in the app
- 🧠 **Automatic Churn Detection**: Detects churn columns based on naming patterns
- ⚙️ **Model Selection**: Auto-trains using the best of Random Forest, XGBoost, and Logistic Regression
- 🧠 **Smart Model Management**:
  - Name your model in the sidebar
  - Reuse existing models or retrain if needed
- 📊 **Interactive Visualizations**:
  - Summary cards
  - Churn distribution charts
  - 3D scatter plots with filters
  - Before vs. After churn comparison
- 📜 **Prediction Logging**: Logs filename, model used, and counts for each session
- 💾 **Model Saving**: Saves models under `models/` with user-defined names

---

## 📦 Project Structure
<pre>
churn-prediction-ml-app/
├── app.py                           # 🚀 Main Streamlit entry point
├── requirements.txt                 # Project dependencies
├── models/
│   └── churn_model.pkl              # Trained model (auto-saved)
├── logs/
│   └── prediction_log.csv           # Prediction + upload logs (auto-created)
├── src/
│   ├── core/
│   │   ├── loader.py                # ✅ FileLoader class (CSV, Excel, JSON)
│   │   ├── model.py                 # ✅ ModelHandler (train, predict, save/load)
│   │   └── logger.py                # ✅ AppLogger (logs filename, rows, etc.)
│   │
│   ├── features/
│   │   ├── cleaner.py               # ✅ DataCleaner (nulls, dedup, churn detection)
│   │   └── engineering.py           # ✅ FeatureEngineer (numeric/categorical)
│   │
│   ├── inference/
│   │   └── predict.py               # ✅ PredictionEngine (runs full pipeline)
│   │
│   └── visualization/
│       └── dashboard.py             # ✅ Visualizer (Streamlit charts & preview)
</pre>
### ✅ Install Requirements

```bash
pip install -r requirements.txt
```
### 🚀 Run the App

```bash
streamlit run app.py
```

### ✅ Example Use Case
    Telecoms identifying at-risk customers

    SaaS analyzing subscription churn

    Edtech platforms predicting student dropout
