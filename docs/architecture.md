# 🏗️ System Architecture

This document provides an overview of the PredictFlex Customer Churn Prediction application's architecture and design patterns.

## 🎯 System Overview

The application follows a modular, layered architecture designed for scalability and maintainability. Built around a Streamlit web interface with a robust ML backend.

## 🏛️ Architecture Layers

### 1. Presentation Layer (UI)
```
┌─────────────────────────────────────┐
│           Streamlit UI              │
│  ┌─────────────┬─────────────────┐  │
│  │   Sidebar   │   Main Content  │  │
│  │  Controls   │   Visualizations│  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
```

**Components:**
- **app.py**: Main Streamlit application
- **Session Management**: State persistence
- **UI Components**: Forms, charts, interactive elements

### 2. Business Logic Layer
```
┌─────────────────────────────────────┐
│         Business Logic              │
│  ┌─────────────┬─────────────────┐  │
│  │ Prediction  │   Visualization │  │
│  │   Engine    │     Engine      │  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
```

**Components:**
- **PredictionEngine**: Orchestrates ML pipeline
- **Visualizer**: Handles visualization logic
- **AppLogger**: Manages logging and tracking

### 3. Data Processing Layer
```
┌─────────────────────────────────────┐
│        Data Processing              │
│  ┌─────────────┬─────────────────┐  │
│  │   Feature   │     Feature     │  │
│  │  Engineering│    Selection    │  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
```

**Components:**
- **DataCleaner**: Data cleaning and preprocessing
- **FeatureEngineer**: Feature transformation
- **FeatureSelector**: Feature selection algorithms

### 4. Model Layer
```
┌─────────────────────────────────────┐
│           Model Layer               │
│  ┌─────────────┬─────────────────┐  │
│  │   Model     │     Model       │  │
│  │  Handler    │   Persistence   │  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
```

**Components:**
- **ModelHandler**: Model training and prediction
- **Model Persistence**: Model saving/loading

## 🔄 Data Flow Architecture

### 1. Data Ingestion Flow
```
User Upload → FileLoader → DataCleaner → FeatureEngineer → ModelHandler
     ↓              ↓            ↓              ↓              ↓
  Streamlit    FileLoader   DataCleaner   FeatureEngineer  ModelHandler
```

### 2. Prediction Flow
```
Input Data → Preprocessing → Feature Engineering → Model Prediction → Results
     ↓            ↓                ↓                ↓              ↓
  Raw Data   Cleaned Data   Engineered Features  Predictions   Visualizations
```

### 3. Model Training Flow
```
Training Data → Data Split → Model Selection → Training → Evaluation → Persistence
      ↓            ↓            ↓            ↓         ↓            ↓
   Raw Data   Train/Test   Algorithm    Training   Metrics    Save Model
```

## 🧩 Component Architecture

### Core Components

#### FileLoader (`src/core/loader.py`)
**Responsibilities:**
- Handle multiple file formats (CSV, Excel, JSON)
- Validate file integrity
- Provide error handling

**Design Pattern:** Factory Pattern

#### ModelHandler (`src/core/model.py`)
**Responsibilities:**
- Manage multiple ML algorithms
- Handle model training and evaluation
- Implement model persistence
- Provide model comparison capabilities

**Design Pattern:** Strategy Pattern

#### PredictionEngine (`src/inference/predict.py`)
**Responsibilities:**
- Orchestrate the complete ML pipeline
- Coordinate between data processing and model training
- Handle both training and inference workflows

**Design Pattern:** Pipeline Pattern

### Feature Engineering Components

#### DataCleaner (`src/features/cleaner.py`)
**Responsibilities:**
- Remove duplicate rows
- Handle missing values
- Detect churn columns automatically
- Ensure data quality

#### FeatureEngineer (`src/features/engineering.py`)
**Responsibilities:**
- Encode categorical variables
- Scale numerical features
- Maintain feature consistency
- Handle unknown categories

#### FeatureSelector (`src/features/feature_selector.py`)
**Responsibilities:**
- Implement multiple feature selection methods
- Reduce dimensionality
- Improve model performance
- Handle feature importance

### Visualization Components

#### Visualizer (`src/visualization/dashboard.py`)
**Responsibilities:**
- Create interactive visualizations
- Display model performance metrics
- Provide data exploration tools
- Generate reports and insights

## 🔧 Design Patterns Used

### 1. Factory Pattern
- **FileLoader**: Creates appropriate data loaders based on file type
- **ModelHandler**: Creates different model types based on configuration

### 2. Strategy Pattern
- **Model Selection**: Different algorithms can be swapped seamlessly
- **Feature Selection**: Multiple feature selection strategies available

### 3. Pipeline Pattern
- **PredictionEngine**: Orchestrates the complete ML pipeline
- **Data Processing**: Sequential data transformation steps

### 4. Observer Pattern
- **Logging**: Automatic logging of predictions and model usage
- **Session Management**: State tracking across user interactions

## 📊 Data Architecture

### Data Storage
```
Project Structure:
├── models/           # Trained model files (.pkl)
├── logs/            # Application logs (.csv)
└── data/            # Sample datasets (optional)
```

### Data Flow Types

#### Training Data Flow
```
Raw Data → Clean → Engineer → Split → Train → Evaluate → Save
```

#### Inference Data Flow
```
New Data → Clean → Engineer → Predict → Visualize
```

#### Batch Processing Flow
```
Multiple Files → Queue → Process → Aggregate → Report
```

## 🔒 Security Architecture

### Data Security
- **Input Validation**: All user inputs are validated
- **File Type Restrictions**: Only allowed file types are processed
- **Memory Management**: Large files are handled efficiently
- **Error Handling**: Comprehensive error handling prevents crashes

### Model Security
- **Model Validation**: Trained models are validated before use
- **Version Control**: Model versions are tracked
- **Backup Strategy**: Models are backed up automatically

## ⚡ Performance Architecture

### Optimization Strategies

#### 1. Caching
```python
@st.cache_data(show_spinner=False, ttl=3600)
def load_dataframe(file):
    # Cached data loading
```

#### 2. Lazy Loading
- Models are loaded only when needed
- Large datasets are sampled for visualization
- Memory-efficient data processing

#### 3. Parallel Processing
- Model training can utilize multiple cores
- Feature engineering is optimized for large datasets

### Scalability Considerations

#### Horizontal Scaling
- Stateless design allows multiple instances
- Session state is minimal and portable
- Database integration ready

#### Vertical Scaling
- Memory-efficient data structures
- Optimized algorithms for large datasets
- Configurable resource limits

## 🔄 State Management

### Session State
```python
# Streamlit session state management
if "step" not in st.session_state:
    st.session_state.step = 1
if "train_df" not in st.session_state:
    st.session_state.train_df = None
```

### Persistent State
- **Model Files**: Saved to disk for reuse
- **Log Files**: Persistent logging across sessions
- **Configuration**: User preferences stored locally

## 🔮 Future Architecture Considerations

### 1. Microservices Architecture
- Separate services for data processing, model training, and inference
- API-based communication between services
- Independent scaling of components

### 2. Cloud-Native Design
- Containerization with Docker
- Kubernetes orchestration
- Cloud storage integration

### 3. Real-time Processing
- Streaming data processing
- Real-time model updates
- Event-driven architecture

---

**Next Steps**: Explore the [API Reference](api_reference.md) for detailed component documentation or [Usage Examples](examples.md) for practical implementation. 