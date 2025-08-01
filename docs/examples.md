# 💡 Usage Examples

This document provides practical examples and use cases for the PredictFlex Customer Churn Prediction application.

## 🚀 Quick Start

### Basic Usage
1. Upload your CSV, Excel, or JSON file
2. System automatically detects churn columns
3. Select model type (or use automatic selection)
4. Train model and view results
5. Explore visualizations and predictions

### Advanced Usage
1. Upload data with custom churn column
2. Choose specific model (Random Forest, XGBoost, etc.)
3. Enable feature selection (RFE, Lasso, Variance Threshold)
4. Train and compare multiple models
5. Analyze feature importance and performance

## 📊 Real-World Examples

### Telecommunications Customer Churn
**Data Structure:**
```csv
customer_id,age,tenure,monthly_charges,total_charges,contract_type,churn
1,25,12,29.85,358.2,Month-to-month,0
2,45,24,56.95,1366.8,One year,1
```

**Steps:**
1. Upload CSV file
2. System detects 'churn' as target
3. Select "Automatic (best accuracy)"
4. Train and analyze results
5. Use insights for retention strategies

### SaaS Subscription Churn
**Data Structure:**
```csv
user_id,subscription_length,monthly_revenue,feature_usage,support_tickets,churn
1,24,99.99,85,2,0
2,6,49.99,45,8,1
```

**Steps:**
1. Upload data
2. Use "XGBoost" for better performance
3. Enable feature selection
4. Train and analyze
5. Develop customer success strategies

## 🔧 Code Examples

### Programmatic Usage
```python
from src.core.loader import FileLoader
from src.inference.predict import PredictionEngine

# Load data
loader = FileLoader(uploaded_file)
df = loader.load()

# Run prediction pipeline
engine = PredictionEngine(model_type="Random Forest")
predictions, enriched_df, model_report = engine.run(df, 'churn')
```

### Custom Feature Engineering
```python
from src.features.cleaner import DataCleaner
from src.features.engineering import FeatureEngineer

# Clean data
cleaner = DataCleaner(df)
cleaned_df = cleaner.clean()

# Engineer features
engineer = FeatureEngineer(target_column='churn')
engineer.fit(cleaned_df.drop('churn', axis=1))
X_transformed = engineer.transform(test_df)
```

### Model Comparison
```python
from src.core.model import ModelHandler

models = ["Random Forest", "XGBoost", "Logistic Regression"]
results = {}

for model_type in models:
    handler = ModelHandler(model_type=model_type)
    model_name, report = handler.train(X, y)
    results[model_name] = report
```

## 📈 Visualization Examples

### Model Performance
```python
from src.visualization.dashboard import Visualizer

visualizer = Visualizer(df, 'prediction')
visualizer.show_model_metrics({
    'Model': 'Random Forest',
    'Accuracy': 85.5,
    'F1 Score': 82.3,
    'ROC AUC': 88.7
})
```

### Churn Analysis
```python
# Compare actual vs predicted churn
visualizer.show_before_after_churn_comparison('actual_churn')

# Show 3D distribution
visualizer.show_3d_distribution()

# Display confusion matrix
visualizer.show_confusion_matrix('actual_churn')
```

## 🎯 Industry Examples

### Banking & Finance
**Use Case**: Credit card customer churn
**Key Features**: Credit score, account balance, transaction frequency
**Model**: XGBoost for high-value customers

### Healthcare
**Use Case**: Patient retention prediction
**Key Features**: Appointment attendance, treatment compliance
**Model**: Logistic Regression for interpretability

### Education
**Use Case**: Student dropout prediction
**Key Features**: Attendance rate, academic performance
**Model**: Random Forest for early intervention

## 🔄 Batch Processing

### Multiple Files
```python
import os
from pathlib import Path

data_dir = Path("data/")
results = {}

for file_path in data_dir.glob("*.csv"):
    loader = FileLoader(file_path)
    df = loader.load()
    
    engine = PredictionEngine()
    predictions, enriched_df, report = engine.run(df, 'churn')
    
    results[file_path.name] = report
```

## 🧪 Testing Examples

### Data Validation
```python
def validate_churn_data(df):
    errors = []
    
    if 'churn' not in df.columns:
        errors.append("Missing churn column")
    
    if df['churn'].nunique() != 2:
        errors.append("Target must be binary (0 or 1)")
    
    return errors
```

### Model Validation
```python
def validate_model_performance(report):
    warnings = []
    
    if report['Accuracy'] < 70:
        warnings.append("Accuracy below 70%")
    
    if report['F1 Score'] < 65:
        warnings.append("F1 score below 65%")
    
    return warnings
```

## 🚀 Deployment Examples

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Streamlit Cloud
```yaml
# .streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"
```

## 📊 Performance Optimization

### Large Datasets
```python
@st.cache_data(show_spinner=False, ttl=3600)
def load_large_dataset(file):
    df = pd.read_csv(file, chunksize=10000)
    return pd.concat(df, ignore_index=True)
```

### Memory Optimization
```python
def optimize_memory(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    return df
```

---

**Next Steps**: Explore the [API Reference](api_reference.md) for detailed component documentation. 