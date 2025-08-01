# 📉 PredictFlex - Customer Churn Prediction App

A comprehensive, user-friendly Streamlit application for predicting customer churn using machine learning. Designed to support both technical and non-technical users with file uploads, real-time predictions, model retraining, and insightful visualizations.

## 🚀 Features

### Core Functionality
- 📁 **Multi-format File Upload**: Support for CSV, Excel, and JSON files
- 🧠 **Automatic Churn Detection**: Intelligent detection of churn columns based on naming patterns
- ⚙️ **Advanced Model Selection**: Auto-trains using multiple algorithms (Random Forest, XGBoost, Logistic Regression, Gradient Boosting, Decision Tree, SVM)
- 🧠 **Smart Model Management**: 
  - Custom model naming and versioning
  - Reuse existing models or retrain as needed
  - Automatic model persistence

### Data Processing
- 🧹 **Intelligent Data Cleaning**: Automatic handling of missing values, duplicates, and data quality issues
- 🔧 **Feature Engineering**: Automatic encoding of categorical variables and scaling of numerical features
- 🎯 **Feature Selection**: Multiple feature selection methods (RFE, Lasso, Variance Threshold)

### Visualization & Analytics
- 📊 **Interactive Dashboards**: Rich visualizations with Plotly and Streamlit
- 📈 **Model Performance Metrics**: Accuracy, F1 Score, ROC AUC with beautiful metric cards
- 🧊 **3D Scatter Plots**: Interactive 3D visualizations with filtering capabilities
- 📊 **Before vs. After Analysis**: Compare actual vs. predicted churn distributions
- 📋 **Confusion Matrix**: Detailed classification performance analysis

### Logging & Monitoring
- 📜 **Comprehensive Logging**: Tracks filename, model used, and prediction counts
- 💾 **Model Persistence**: Automatic model saving with user-defined names
- 📊 **Session Management**: Maintains state across Streamlit sessions

## 📦 Project Structure

```
predictflex/
├── app.py                           # 🚀 Main Streamlit application
├── requirements.txt                 # 📦 Project dependencies
├── readme.md                       # 📖 Project documentation
├── docs/                           # 📚 Comprehensive documentation
│   ├── setup.md                    # 🛠️ Detailed setup guide
│   ├── architecture.md             # 🏗️ System architecture
│   ├── api_reference.md            # 🔧 API documentation
│   └── examples.md                 # 💡 Usage examples
├── models/                         # 💾 Trained model storage
│   └── churn_model.pkl            # Default model file
├── logs/                          # 📝 Application logs
│   └── prediction_log.csv         # Prediction tracking
└── src/                           # 🔧 Source code
    ├── core/                      # 🧠 Core functionality
    │   ├── loader.py              # 📁 File loading utilities
    │   ├── model.py               # 🤖 Model training and prediction
    │   ├── logger.py              # 📝 Logging functionality
    │   └── pipeline.py            # 🔄 Data processing pipeline
    ├── features/                  # 🔧 Feature engineering
    │   ├── cleaner.py             # 🧹 Data cleaning utilities
    │   ├── engineering.py         # 🔧 Feature engineering
    │   └── feature_selector.py    # 🎯 Feature selection methods
    ├── inference/                 # 🔮 Prediction engine
    │   └── predict.py             # 🎯 Prediction pipeline
    └── visualization/             # 📊 Visualization components
        └── dashboard.py           # 📈 Dashboard components
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
cd predictflex
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically open in your default browser

### Detailed Setup Guide
For comprehensive setup instructions, see [docs/setup.md](docs/setup.md).

## 🎯 Usage Examples

### Basic Usage
1. **Upload Data**: Upload your CSV, Excel, or JSON file containing customer data
2. **Configure Model**: Choose your preferred model type or use automatic selection
3. **Train Model**: The system will automatically detect churn columns and train the model
4. **View Results**: Explore predictions, visualizations, and performance metrics

### Advanced Usage
- **Custom Feature Selection**: Use RFE, Lasso, or Variance Threshold methods
- **Model Comparison**: Compare multiple algorithms side-by-side
- **Batch Processing**: Process multiple files in sequence
- **Model Persistence**: Save and reuse trained models

### Example Use Cases
- **Telecommunications**: Identify at-risk customers for retention campaigns
- **SaaS Platforms**: Predict subscription churn and optimize pricing strategies
- **E-commerce**: Analyze customer behavior and improve retention
- **Education**: Predict student dropout rates and implement interventions

## 🔧 Configuration

### Model Types Available
- **Automatic (best accuracy)**: Automatically selects the best performing model
- **Random Forest**: Ensemble method with high accuracy
- **XGBoost**: Gradient boosting with excellent performance
- **Logistic Regression**: Linear model with good interpretability
- **Gradient Boosting**: Another ensemble method
- **Decision Tree**: Simple, interpretable model
- **Support Vector Machine**: Good for complex patterns

### Feature Selection Methods
- **RFE (Recursive Feature Elimination)**: Removes least important features
- **Lasso**: Uses L1 regularization for feature selection
- **Variance Threshold**: Removes low-variance features

## 📊 Performance Metrics

The application provides comprehensive model evaluation metrics:
- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve for classification performance

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the [documentation](docs/)
- Review the [examples](docs/examples.md)
- Open an issue on the repository

---

**PredictFlex - Built with ❤️ using Streamlit, Scikit-learn, and XGBoost**
