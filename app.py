# app.py

import streamlit as st
import os
from src.core.loader import FileLoader
from src.inference.predict import PredictionEngine
from src.visualization.dashboard import Visualizer
from src.core.logger import AppLogger

st.set_page_config(page_title="Churn Prediction", layout="wide",initial_sidebar_state="auto")
st.title("📉 Churn Prediction App")

# 1️⃣ Sidebar Inputs
st.sidebar.header("Upload & Model Options")

uploaded_file = st.sidebar.file_uploader("Upload your data", type=["csv", "xlsx", "json"])
model_name = st.sidebar.text_input("Model Name", value="churn_model",max_chars=50)
model_filename = f"models/{model_name}.pkl"

retrain = st.sidebar.checkbox("Retrain model if not found", value=True)

# 2️⃣ If File Uploaded
if uploaded_file:
    st.success(f"✅ File `{uploaded_file.name}` uploaded successfully!")

    try:
        loader = FileLoader(uploaded_file)
        df = loader.load()
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head(50))
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    # 3️⃣ Prediction Engine
    st.info("Running churn prediction...")

    try:
        engine = PredictionEngine(model_path=model_filename)
        prediction_df, enriched_df = engine.run(df, train_if_no_model=retrain)
        st.success("✅ Prediction completed!")

        # Logging
        logger = AppLogger()
        logger.log_prediction(
            filename=uploaded_file.name,
            num_rows=len(df),
            num_predictions=len(prediction_df),
            model_version=model_name
        )

        # Show visuals
        viz = Visualizer(enriched_df)
        viz.show_summary_cards()
        # viz.show_churn_distribution()
        viz.show_before_after_churn_comparison(actual_col=engine.churn_column)
        viz.show_3d_distribution()
        viz.show_filter_and_preview(actual_col=engine.churn_column)


    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
