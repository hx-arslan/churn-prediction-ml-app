# 🔧 API Reference

This document provides API documentation for the PredictFlex Customer Churn Prediction application components.

## 🧠 Core Components

### FileLoader (`src/core/loader.py`)

**Purpose**: Handles file loading for multiple formats

```python
class FileLoader:
    def __init__(self, uploaded_file)
    def load(self) -> pd.DataFrame
```

**Supported Formats**: CSV, Excel (.xlsx, .xls), JSON

**Example**:
```python
from src.core.loader import FileLoader
loader = FileLoader(uploaded_file)
df = loader.load()
```

### ModelHandler (`src/core/model.py`)

**Purpose**: Manages model training, prediction, and persistence

```python
class ModelHandler:
    def __init__(self, model_path='models/churn_model.pkl', model_type="Automatic (best accuracy)")
    def train(self, X, y) -> tuple
    def predict(self, X) -> np.array
    def save(self) -> None
    def load(self) -> None
```

**Available Model Types**:
- `"Automatic (best accuracy)"` - Auto-selects best model
- `"Random Forest"` - Random Forest Classifier
- `"XGBoost"` - XGBoost Classifier
- `"Logistic Regression"` - Logistic Regression
- `"Gradient Boosting"` - Gradient Boosting Classifier
- `"Decision Tree"` - Decision Tree Classifier
- `"Support Vector Machine"` - SVM Classifier

**Example**:
```python
from src.core.model import ModelHandler
handler = ModelHandler(model_type="Random Forest")
model_name, report = handler.train(X, y)
predictions = handler.predict(X_test)
```

### AppLogger (`src/core/logger.py`)

**Purpose**: Handles application logging and tracking

```python
class AppLogger:
    def __init__(self, log_file='logs/prediction_log.csv')
    def log_prediction(self, filename, model_name, row_count, prediction_count)
    def get_logs(self) -> pd.DataFrame
```

**Example**:
```python
from src.core.logger import AppLogger
logger = AppLogger()
logger.log_prediction("data.csv", "Random Forest", 1000, 1000)
```

## 🔧 Feature Engineering

### DataCleaner (`src/features/cleaner.py`)

**Purpose**: Handles data cleaning and preprocessing

```python
class DataCleaner:
    def __init__(self, df: pd.DataFrame)
    def detect_churn_column(self) -> str
    def clean(self) -> pd.DataFrame
```

**Operations**:
- Removes columns with all null values
- Removes duplicate rows
- Fills missing values using forward/backward fill
- Detects churn columns automatically

**Example**:
```python
from src.features.cleaner import DataCleaner
cleaner = DataCleaner(df)
churn_col = cleaner.detect_churn_column()
cleaned_df = cleaner.clean()
```

### FeatureEngineer (`src/features/engineering.py`)

**Purpose**: Handles feature engineering and transformation

```python
class FeatureEngineer:
    def __init__(self, target_column: str = None)
    def fit(self, df: pd.DataFrame) -> None
    def transform(self, df: pd.DataFrame) -> pd.DataFrame
```

**Operations**:
- Scales numerical features using StandardScaler
- Encodes categorical features using OneHotEncoder
- Maintains feature consistency across datasets

**Example**:
```python
from src.features.engineering import FeatureEngineer
engineer = FeatureEngineer(target_column='churn')
engineer.fit(training_df)
X_transformed = engineer.transform(test_df)
```

### FeatureSelector (`src/features/feature_selector.py`)

**Purpose**: Implements feature selection algorithms

```python
class FeatureSelector:
    def __init__(self, method='RFE', num_features=None, threshold=0.01)
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame
    def get_selected_features(self) -> list
```

**Available Methods**:
- `'RFE'` - Recursive Feature Elimination
- `'Lasso'` - L1 regularization-based selection
- `'VarianceThreshold'` - Variance-based selection

**Example**:
```python
from src.features.feature_selector import FeatureSelector
selector = FeatureSelector(method='RFE', num_features=10)
X_selected = selector.select_features(X, y)
```

## 🔮 Inference Engine

### PredictionEngine (`src/inference/predict.py`)

**Purpose**: Orchestrates the complete ML pipeline

```python
class PredictionEngine:
    def __init__(self, model_path='models/churn_model.pkl', model_type="Automatic (best accuracy)")
    def preprocess(self, df, target_column, feature_columns) -> tuple
    def run(self, df, target_column, feature_columns, train_if_no_model=True, feature_selection_method=None) -> tuple
    def evaluate_on_test_data(self, test_df, target_column, feature_columns) -> dict
```

**Pipeline Steps**:
1. Data cleaning and preprocessing
2. Feature engineering
3. Feature selection (optional)
4. Model training/prediction
5. Results evaluation

**Example**:
```python
from src.inference.predict import PredictionEngine
engine = PredictionEngine()
predictions, enriched_df, model_report = engine.run(df, 'churn')
```

## 📊 Visualization

### Visualizer (`src/visualization/dashboard.py`)

**Purpose**: Handles all visualization and dashboard components

```python
class Visualizer:
    def __init__(self, df: pd.DataFrame, prediction_col: str = 'prediction')
    def show_model_metrics(self, metrics) -> None
    def show_before_after_churn_comparison(self, actual_col: str = None) -> None
    def show_3d_distribution(self) -> None
    def show_confusion_matrix(self, actual_col: str = None) -> None
    def show_feature_importance(self, feature_columns=None) -> None
```

**Visualization Types**:
- Model performance metrics cards
- Before vs. after churn comparison
- Interactive 3D scatter plots
- Confusion matrix
- Feature importance plots
- Prediction distribution charts

**Example**:
```python
from src.visualization.dashboard import Visualizer
visualizer = Visualizer(df, 'prediction')
visualizer.show_model_metrics(metrics)
visualizer.show_confusion_matrix('actual_churn')
```

## 📊 Data Types

### Input Data Requirements

```python
# Required DataFrame structure
{
    'customer_id': int,      # Unique customer identifier
    'feature_1': float,      # Numerical feature
    'feature_2': str,        # Categorical feature
    'churn': int            # Target variable (0 or 1)
}
```

### Output Data Formats

```python
# Prediction Results
{
    'prediction': np.array,  # Predicted values (0 or 1)
    'probability': np.array, # Prediction probabilities
    'confidence': float      # Model confidence score
}

# Model Metrics
{
    'Model': str,           # Model name
    'Accuracy': float,      # Accuracy percentage
    'F1 Score': float,      # F1 score percentage
    'ROC AUC': float        # ROC AUC percentage
}
```

## ⚠️ Error Handling

### Common Exceptions

```python
# FileLoader Errors
try:
    loader = FileLoader(uploaded_file)
    df = loader.load()
except ValueError as e:
    st.error(f"Unsupported file format: {e}")
except RuntimeError as e:
    st.error(f"File loading error: {e}")

# ModelHandler Errors
try:
    handler = ModelHandler()
    model_name, report = handler.train(X, y)
except Exception as e:
    st.error(f"Model training failed: {e}")

# FeatureEngineer Errors
try:
    engineer = FeatureEngineer(target_column='churn')
    engineer.fit(df)
    X_transformed = engineer.transform(df)
except ValueError as e:
    st.error(f"Feature engineering error: {e}")
```

### Data Validation

```python
def validate_data(df):
    """Validate input data requirements"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if 'churn' not in df.columns:
        raise ValueError("Target column 'churn' not found")
    
    if df['churn'].nunique() != 2:
        raise ValueError("Target column must be binary (0 or 1)")
    
    return True
```

---

**Next Steps**: Explore [Usage Examples](examples.md) for practical implementation scenarios. 