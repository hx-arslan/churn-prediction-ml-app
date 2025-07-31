import streamlit as st
import os
from src.core.loader import FileLoader
from src.inference.predict import PredictionEngine
from src.visualization.dashboard import Visualizer
from src.core.logger import AppLogger

# Must be the first Streamlit command
st.set_page_config(page_title="PredictFlex", layout="wide", initial_sidebar_state="auto")

@st.cache_data(show_spinner=False, ttl=3600)
def load_dataframe(file):
    loader = FileLoader(file)
    return loader.load()

@st.cache_data(show_spinner=False, ttl=3600)
def get_data_preview(df, max_rows=10):
    """Get a preview of the dataframe with limited rows to prevent freezing"""
    # For very large datasets, sample instead of head to prevent UI freezing
    if len(df) > 50000:
        return df.sample(n=min(max_rows, 1000), random_state=42)
    else:
        return df.head(max_rows)

@st.cache_data(show_spinner=False, ttl=3600)
def get_target_stats(df, target_col):
    """Get target statistics efficiently"""
    if target_col not in df.columns:
        return None, 0, 0
    # Limit the data size to prevent UI freezing
    if len(df) > 10000:
        sample_df = df.sample(n=10000, random_state=42)
    else:
        sample_df = df
    target_counts = sample_df[target_col].value_counts()
    unique_values = df[target_col].nunique()
    missing_values = df[target_col].isnull().sum()
    return target_counts, unique_values, missing_values

@st.cache_data(show_spinner=False, ttl=3600)
def get_feature_stats(df, feature_cols):
    """Get feature statistics efficiently"""
    if not feature_cols:
        return None
    return df[feature_cols].describe()

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "test_df" not in st.session_state:
    st.session_state.test_df = None
if "uploaded_train_filename" not in st.session_state:
    st.session_state.uploaded_train_filename = ""
if "uploaded_test_filename" not in st.session_state:
    st.session_state.uploaded_test_filename = ""
if "file_mode" not in st.session_state:
    st.session_state.file_mode = "single"


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1

# Custom CSS for better styling
st.markdown("""
<style>
    /* 1. HEADER STYLING */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* 2. METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* 3. INFO BOXES */
    .info-box {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* 4. STEP HEADERS */
    .step-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* 5. DATA PREVIEW CARDS */
    .data-preview-card {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* 6. SIDEBAR TEXT - LIGHT ON DARK BACKGROUND */
    .css-1d391kg {
        color: #ecf0f1 !important;
    }
    
    .css-1d391kg h3 {
        color: #ecf0f1 !important;
    }
    
    .css-1d391kg label {
        color: #ecf0f1 !important;
    }
    
    /* 7. FORM ELEMENTS - DARK TEXT ON WHITE BACKGROUND */
    .stRadio > div {
        background: black;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #bdc3c7;
        color: #2c3e50 !important;
    }
    
    .stRadio > div > div > div > div > div > div > div > div > label {
        color: #2c3e50 !important;
    }
    
    .stCheckbox > div {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #bdc3c7;
        color: #2c3e50 !important;
    }
    
    .stCheckbox > div > div > div > div > div > div > div > div > label {
        color: #2c3e50 !important;
    }
    
    .stTextInput > div > div > input {
        background: white;
        border-radius: 8px;
        border: 2px solid #bdc3c7;
        color: #2c3e50 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* 8. FILE UPLOADER - LIGHT BACKGROUND WITH DARK TEXT */
    .stFileUploader > div {
        background: black;
        border-radius: 8px;
        border: 2px dashed #bdc3c7;
        padding: 1rem;
        color: white !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #3498db;
        background: black;
    }
    
    /* 9. BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f5f8b 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* 10. DATA TABLES */
    .stDataFrame {
        background: white;
        border-radius: 8px;
        border: 1px solid #bdc3c7;
        overflow: hidden;
        color: #2c3e50 !important;
    }
    
    .stDataFrame th, .stDataFrame td {
        color: #2c3e50 !important;
    }
    
    /* 11. PROGRESS BARS */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
    }
    
    /* 12. SPINNERS */
    .stSpinner > div {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
    }
    
    /* 13. ALERTS */
    .stAlert {
        color: #ecf0f1 !important;
    }
    
    .stSuccess {
        color: #27ae60 !important;
    }
    
    .stWarning {
        color: #f39c12 !important;
    }
    
    .stError {
        color: #e74c3c !important;
    }
    
    /* 14. MAIN CONTENT */
    .main .block-container {
        color: #ecf0f1 !important;
    }
    
    .main .block-container * {
        color: #ecf0f1 !important;
    }
    
    /* 15. UNIVERSAL OVERRIDE FOR GRADIENT CARDS */
    .metric-card *, .info-box *, .success-box *, .step-header *, .warning-box *, .error-box *, .data-preview-card * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header"><h1>🎯 PredictFlex - Advanced ML Predictor</h1></div>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("### 📁 Data Upload Options")
file_mode = st.sidebar.radio(
    "Select File Mode:",
    options=["Single File (Training Only)", "Dual Files (Training + Testing)"],
    index=0 if st.session_state.file_mode == "single" else 1,
    key="file_mode_radio"
)

# Update session state
st.session_state.file_mode = "single" if file_mode == "Single File (Training Only)" else "dual"

st.sidebar.markdown("### 📊 Model Configuration")
model_name = st.sidebar.text_input("Model Name", value="churn_model", max_chars=50)
model_filename = f"models/{model_name}.pkl"
retrain = st.sidebar.checkbox("Retrain model if not found", value=True)

# File upload based on mode
if st.session_state.file_mode == "single":
    st.sidebar.markdown("### 📤 Upload Training Data")
    uploaded_train_file = st.sidebar.file_uploader(
        "Upload your dataset", 
        type=["csv", "xlsx", "json"],
        help="Upload your training dataset (CSV, Excel, or JSON format)"
    )
    uploaded_test_file = None
else:
    st.sidebar.markdown("### 📤 Upload Data Files")
    uploaded_train_file = st.sidebar.file_uploader(
        "Upload Training Data (Required)", 
        type=["csv", "xlsx", "json"],
        help="Upload your training dataset"
    )
    uploaded_test_file = st.sidebar.file_uploader(
        "Upload Testing Data (Optional)", 
        type=["csv", "xlsx", "json"],
        help="Upload your testing dataset (optional)"
    )

# Submit button with validation
submit_disabled = not uploaded_train_file or not model_name
submit_text = "Submit" if not submit_disabled else "Please upload training data and enter model name"

# Add loading state to session state
if "loading" not in st.session_state:
    st.session_state.loading = False

if st.sidebar.button(submit_text, disabled=submit_disabled, type="primary"):
    st.session_state.loading = True
    st.session_state.submitted = True
    
    # Show loading message with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load training data
        status_text.text("🔄 Loading training data...")
        progress_bar.progress(25)
        
        if uploaded_train_file.name != st.session_state.uploaded_train_filename:
            st.session_state.train_df = load_dataframe(uploaded_train_file)
            st.session_state.uploaded_train_filename = uploaded_train_file.name
        
        progress_bar.progress(50)
        status_text.text("🔄 Processing data...")
        
        # Load testing data if provided
        if uploaded_test_file and uploaded_test_file.name != st.session_state.uploaded_test_filename:
            status_text.text("🔄 Loading testing data...")
            st.session_state.test_df = load_dataframe(uploaded_test_file)
            st.session_state.uploaded_test_filename = uploaded_test_file.name
        elif not uploaded_test_file:
            st.session_state.test_df = None
            st.session_state.uploaded_test_filename = ""
        
        progress_bar.progress(75)
        status_text.text("🔄 Preparing interface...")
        
        st.session_state.step = 1  # reset to step 1 on new upload
        
        progress_bar.progress(100)
        status_text.text("✅ Data loaded successfully!")
        
        st.session_state.loading = False
        st.success("✅ Data loaded successfully! Ready for analysis.")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.session_state.loading = False
    finally:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

# Main workflow
if st.session_state.submitted and st.session_state.train_df is not None:
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    
    # Data preview with enhanced styling
    st.markdown('<div class="step-header"><h3>📊 Data Preview</h3></div>', unsafe_allow_html=True)
    
    # Add loading for data preview with optimized functions
    progress_placeholder = st.empty()
    progress_placeholder.progress(0)
    progress_placeholder.text("📊 Loading data preview...")
    
    try:
        col1, col2 = st.columns(2)
        
        progress_placeholder.progress(25)
        progress_placeholder.text("📊 Loading training data preview...")
        
        with col1:
            st.markdown("**📈 Training Data:**")
            train_preview = get_data_preview(train_df, max_rows=10)
            st.dataframe(train_preview, use_container_width=True)
            st.markdown(f"<div class='metric-card'>📏 Shape: {train_df.shape[0]} rows × {train_df.shape[1]} columns</div>", unsafe_allow_html=True)
        
        progress_placeholder.progress(50)
        progress_placeholder.text("📊 Loading testing data preview...")
        
        if test_df is not None:
            with col2:
                st.markdown("**🧪 Testing Data:**")
                test_preview = get_data_preview(test_df, max_rows=10)
                st.dataframe(test_preview, use_container_width=True)
                st.markdown(f"<div class='metric-card'>📏 Shape: {test_df.shape[0]} rows × {test_df.shape[1]} columns</div>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown('<div class="info-box">ℹ️ No testing data provided. Model will be evaluated on training data split.</div>', unsafe_allow_html=True)
        
        progress_placeholder.progress(100)
        progress_placeholder.text("✅ Data preview loaded!")
        
    except Exception as e:
        st.error(f"Error loading data preview: {str(e)}")
    finally:
        progress_placeholder.empty()

    # Step 1: Select Target Column
    if st.session_state.step == 1:
        st.markdown('<div class="step-header"><h3>🎯 Step 1: Select Target Column</h3></div>', unsafe_allow_html=True)

        target_column = st.selectbox(
            "Select the Target Column (Label)", 
            train_df.columns,
            help="Choose the column that contains your target variable (e.g., churn, fraud, etc.)"
        )

        st.session_state.target_col = target_column
        
        # Show target distribution with lazy loading
        if target_column in train_df.columns:
            # Create placeholder containers
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Target Distribution:**")
                chart_placeholder = st.empty()
                
            with col2:
                st.write("**Target Statistics:**")
                stats_placeholder = st.empty()
            
            # Load data in background with progress
            progress_placeholder = st.empty()
            progress_placeholder.progress(0)
            
            # Process in chunks to prevent UI freezing
            try:
                progress_placeholder.progress(25)
                progress_placeholder.text("📊 Calculating target statistics...")
                
                target_counts, unique_values, missing_values = get_target_stats(train_df, target_column)
                
                progress_placeholder.progress(75)
                progress_placeholder.text("📊 Generating chart...")
                
                if target_counts is not None:
                    chart_placeholder.bar_chart(target_counts)
                
                progress_placeholder.progress(100)
                progress_placeholder.text("✅ Target analysis complete!")
                
                stats_placeholder.write(f"Unique values: {unique_values}")
                stats_placeholder.write(f"Missing values: {missing_values}")
                
            except Exception as e:
                st.error(f"Error generating target distribution: {str(e)}")
            finally:
                # Clear progress indicator
                progress_placeholder.empty()
        
        st.button("Next ➡️", on_click=next_step, type="primary")

    # Step 2: Select Feature Selection Method
    elif st.session_state.step == 2:
        st.markdown('<div class="step-header"><h3>🔍 Step 2: Select Feature Selection Method</h3></div>', unsafe_allow_html=True)

        methods = ["None", "RFE", "Lasso", "VarianceThreshold"]
        selected_methods = st.multiselect(
            "Choose one or more feature selection methods:", 
            methods,
            help="Feature selection helps improve model performance by selecting the most important features"
        )
        selected_methods = [m for m in selected_methods if m != "None"]

        st.session_state.feature_selection_method = selected_methods if selected_methods else None

        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step)
        with col2:
            st.button("Next ➡️", on_click=next_step, type="primary")

    # Step 3: Select Feature Columns
    elif st.session_state.step == 3:
        st.markdown('<div class="step-header"><h3>⚙️ Step 3: Select Feature Columns</h3></div>', unsafe_allow_html=True)

        if "target_col" not in st.session_state:
            st.error("Please go back and select a target column.")
            if st.button("⬅️ Back"):
                prev_step()
        else:
            target_column = st.session_state.target_col
            feature_columns = st.multiselect(
                "Select feature columns (or leave blank to use all except target)",
                [col for col in train_df.columns if col != target_column],
                help="Select the features you want to use for training. Leave blank to use all columns except the target."
            )

            st.session_state.feature_cols = feature_columns
            
            # Show feature statistics with lazy loading
            if feature_columns:
                st.write("**Selected Features Statistics:**")
                stats_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                try:
                    progress_placeholder.progress(0)
                    progress_placeholder.text("📊 Calculating feature statistics...")
                    
                    feature_stats = get_feature_stats(train_df, feature_columns)
                    
                    progress_placeholder.progress(100)
                    progress_placeholder.text("✅ Feature statistics complete!")
                    
                    if feature_stats is not None:
                        stats_placeholder.dataframe(feature_stats, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error calculating feature statistics: {str(e)}")
                finally:
                    progress_placeholder.empty()
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("⬅️ Back", on_click=prev_step)
            with col2:
                st.button("Next ➡️", on_click=next_step, type="primary")

    # Step 4: Choose Model
    elif st.session_state.step == 4:
        st.markdown('<div class="step-header"><h3>🤖 Step 4: Choose a Model</h3></div>', unsafe_allow_html=True)

        # Enhanced model options with descriptions
        model_options = {
            "Automatic (best accuracy)": "Let the system choose the best performing model automatically",
            "Random Forest": "Ensemble method, good for complex patterns, handles non-linear relationships",
            "XGBoost": "Advanced gradient boosting, excellent performance, handles missing values",
            "Logistic Regression": "Simple, interpretable, good baseline model",
            "Gradient Boosting": "Sequential ensemble, good performance, handles various data types",
            "Decision Tree": "Simple, interpretable, can handle non-linear relationships",
            "Support Vector Machine": "Good for high-dimensional data, robust to overfitting"
        }

        model_option = st.radio(
            "Select a model for training:",
            options=list(model_options.keys()),
            format_func=lambda x: f"{x} - {model_options[x]}"
        )

        st.session_state.model_choice = model_option
        
        # Show model recommendations
        if model_option != "Automatic (best accuracy)":
            st.info(f"💡 **Recommendation:** {model_options[model_option]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step)
        with col2:
            st.button("Submit Model Choice ✅", on_click=next_step, type="primary")

    # Step 5: Run Prediction
    elif st.session_state.step == 5:
        st.markdown('<div class="step-header"><h3>🚀 Step 5: Running Prediction & Evaluation</h3></div>', unsafe_allow_html=True)

        target_column = st.session_state.target_col
        feature_columns = st.session_state.feature_cols
        model_choice = st.session_state.model_choice
        feature_selection_method = st.session_state.feature_selection_method

        try:
            with st.spinner("🔄 Training model..."):
                engine = PredictionEngine(model_path=model_filename, model_type=model_choice)
                
                # Train model and get training metrics
                train_prediction_df, train_enriched_df, train_model_report = engine.run(
                    train_df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    train_if_no_model=retrain,
                    feature_selection_method=feature_selection_method
                )

            st.markdown('<div class="success-box">✅ Model training completed successfully!</div>', unsafe_allow_html=True)

            # If testing data is provided, evaluate on test data
            test_prediction_df = None
            test_enriched_df = None
            test_model_report = None
            
            if test_df is not None:
                with st.spinner("🔄 Evaluating model on testing data..."):
                    test_prediction_df, test_enriched_df, test_model_report = engine.evaluate_on_test_data(
                        test_df,
                        target_column=target_column,
                        feature_columns=feature_columns
                    )
                st.markdown('<div class="success-box">✅ Testing evaluation completed!</div>', unsafe_allow_html=True)

            # Log the prediction
            logger = AppLogger()
            logger.log_prediction(
                filename=uploaded_train_file.name,
                num_rows=len(train_df),
                num_predictions=len(train_prediction_df),
                model_version=model_name
            )

            # Show results based on available data
            if test_df is not None:
                # Show both training and testing metrics
                st.markdown('<div class="step-header"><h3>📊 Model Performance Comparison</h3></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📈 Training Metrics:**")
                    viz_train = Visualizer(train_enriched_df)
                    viz_train.show_model_metrics(train_model_report)
                
                with col2:
                    st.markdown("**🧪 Testing Metrics:**")
                    viz_test = Visualizer(test_enriched_df)
                    viz_test.show_model_metrics(test_model_report)
                
                # Show enhanced visualizations for testing data
                viz_test.show_summary_cards()
                viz_test.show_before_after_churn_comparison(actual_col=engine.churn_column)
                viz_test.show_confusion_matrix(actual_col=engine.churn_column)
                viz_test.show_feature_importance(feature_columns=feature_columns)
                viz_test.show_prediction_distribution()
                viz_test.show_3d_distribution()
                viz_test.show_filter_and_preview(actual_col=engine.churn_column)
                
            else:
                # Show only training metrics
                st.markdown('<div class="step-header"><h3>📊 Training Metrics</h3></div>', unsafe_allow_html=True)
                viz = Visualizer(train_enriched_df)
                viz.show_model_metrics(train_model_report)
                viz.show_summary_cards()
                viz.show_before_after_churn_comparison(actual_col=engine.churn_column)
                viz.show_confusion_matrix(actual_col=engine.churn_column)
                viz.show_feature_importance(feature_columns=feature_columns)
                viz.show_prediction_distribution()
                viz.show_3d_distribution()
                viz.show_filter_and_preview(actual_col=engine.churn_column)

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**")
            st.info("• Check if your data has the required columns")
            st.info("• Ensure your target column has valid values")
            st.info("• Try different feature selection methods")
            st.info("• Consider using 'Automatic' model selection")

st.markdown("---")
st.markdown('<div class="step-header"><h3>🔄 Start Over</h3></div>', unsafe_allow_html=True)

if st.button("🔄 Reset App", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()