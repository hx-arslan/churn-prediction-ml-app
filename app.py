import streamlit as st
import os
from src.core.loader import FileLoader
from src.inference.predict import PredictionEngine
from src.visualization.dashboard import Visualizer
from src.core.logger import AppLogger

@st.cache_data(show_spinner=False)
def load_dataframe(file):
    loader = FileLoader(file)
    return loader.load()

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "df" not in st.session_state:
    st.session_state.df = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

st.set_page_config(page_title="Churn Prediction", layout="wide", initial_sidebar_state="auto")
st.title("📉 Churn Prediction App")

# Sidebar inputs
st.sidebar.header("Upload & Model Options")
uploaded_file = st.sidebar.file_uploader("Upload your data", type=["csv", "xlsx", "json"])
model_name = st.sidebar.text_input("Model Name", value="churn_model", max_chars=50)
model_filename = f"models/{model_name}.pkl"
retrain = st.sidebar.checkbox("Retrain model if not found", value=True)

# On submit, load file once and store in session
if st.sidebar.button("Submit") and uploaded_file and model_name:
    st.session_state.submitted = True
    if uploaded_file.name != st.session_state.uploaded_filename:
        st.session_state.df = load_dataframe(uploaded_file)
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.step = 1  # reset to step 1 on new upload

# Main workflow
if st.session_state.submitted and st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head(50))

    # Step 1: Select Target Column
    if st.session_state.step == 1:
        st.subheader("🧮 Step 1: Select Target Column")

        target_column = st.selectbox("🎯 Select the Target Column (Label)", df.columns)

        st.session_state.target_col = target_column
        st.button("Next ➡️", on_click=next_step)            

    # Step 2: Select Feature Columns
    elif st.session_state.step == 2:
        st.subheader("🧩 Step 2: Select Feature Columns")

        if "target_col" not in st.session_state:
            st.error("Please go back and select a target column.")
            if st.button("⬅️ Back"):
                prev_step()
        else:
            target_column = st.session_state.target_col
            feature_columns = st.multiselect(
                "Select feature columns (or leave blank to use all except target)",
                [col for col in df.columns if col != target_column]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.button("⬅️ Back", on_click=prev_step)
            with col2:
                st.session_state.feature_cols = feature_columns
                st.button("Next ➡️", on_click=next_step)

    # Step 3: Choose Model
    elif st.session_state.step == 3:
        st.subheader("🤖 Step 3: Choose a Model")

        model_option = st.radio(
            "Select a model for training",
            options=[
    "Automatic (best accuracy)",
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "XGBoost",
    "Gradient Boosting",
    "AdaBoost",
    "Ridge Classifier",
    "Gaussian Naive Bayes",
    "K-Nearest Neighbors",
    "Support Vector Machine"
]

        )

        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=prev_step)
        with col2:
            st.session_state.model_choice = model_option
            st.button("Submit Model Choice ✅", on_click=next_step)

    # Step 4: Run Prediction
    elif st.session_state.step == 4:
        st.subheader("🚀 Step 4: Running Prediction...")

        target_column = st.session_state.target_col
        feature_columns = st.session_state.feature_cols
        model_choice = st.session_state.model_choice

        try:
            X = df.drop(columns=[target_column]) if not feature_columns else df[feature_columns]
            y = df[target_column]

            engine = PredictionEngine(model_path=model_filename, model_type=model_choice)
            prediction_df, enriched_df = engine.run(df, target_column=target_column, feature_columns=feature_columns, train_if_no_model=retrain)

            st.success("✅ Prediction completed!")

            logger = AppLogger()
            logger.log_prediction(
                filename=uploaded_file.name,
                num_rows=len(df),
                num_predictions=len(prediction_df),
                model_version=model_name
            )

            viz = Visualizer(enriched_df)
            viz.show_summary_cards()
            viz.show_before_after_churn_comparison(actual_col=engine.churn_column)
            viz.show_3d_distribution()
            viz.show_filter_and_preview(actual_col=engine.churn_column)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            
st.markdown("---")
st.subheader("🔄 Start Over")

if st.button("🔁 Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


