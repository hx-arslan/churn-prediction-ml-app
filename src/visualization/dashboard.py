# src/visualization/dashboard.py

import streamlit as st
import plotly.express as px
import pandas as pd

class Visualizer:
    def __init__(self, df: pd.DataFrame, prediction_col: str = 'churn_prediction'):
        self.df = df
        self.prediction_col = prediction_col

    def show_before_after_churn_comparison(self, actual_col: str = None):
        st.subheader("📊 Before vs. After: Churn Comparison")

        data = []

        # Before (actual churn)
        if actual_col and actual_col in self.df.columns:
            actual_counts = self.df[actual_col].value_counts(normalize=True) * 100
            for label, pct in actual_counts.items():
                data.append({"Source": "Actual", "Churn": str(label), "Percentage": pct})
        else:
            st.info("No actual churn column detected for 'before' comparison.")

        # After (predicted churn)
        if self.prediction_col in self.df.columns:
            predicted_counts = self.df[self.prediction_col].value_counts(normalize=True) * 100
            for label, pct in predicted_counts.items():
                data.append({"Source": "Predicted", "Churn": str(label), "Percentage": pct})

        if data:
            df_plot = pd.DataFrame(data)
            fig = px.bar(
                df_plot,
                x="Churn",
                y="Percentage",
                color="Source",
                barmode="group",
                title="Before vs. After: Churn Distribution (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_3d_distribution(self):
        st.subheader("🧊 Churn Insights (3D Scatter Plot)")

        # Add a filter for prediction value
        selected = st.selectbox("Filter 3D plot by prediction", options=["All", "Churned", "Not Churned"])
        if selected == "Churned":
            data = self.df[self.df[self.prediction_col] == 1]
        elif selected == "Not Churned":
            data = self.df[self.df[self.prediction_col] == 0]
        else:
            data = self.df

        numeric_cols = data.select_dtypes(include='number').drop(columns=[self.prediction_col], errors='ignore')

        if len(numeric_cols.columns) >= 3:
            fig = px.scatter_3d(
                data,
                x=numeric_cols.columns[0],
                y=numeric_cols.columns[1],
                z=numeric_cols.columns[2],
                color=self.prediction_col,
                title="3D Feature Scatter by Churn"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for 3D plot.")

    def show_summary_cards(self):
        st.subheader("📌 Summary Stats")
        col1, col2, col3 = st.columns(3)

        total = len(self.df)
        churned = int((self.df[self.prediction_col] == 1).sum())
        not_churned = total - churned

        col1.metric("Total Customers", total)
        col2.metric("Churned", churned)
        col3.metric("Not Churned", not_churned)

    def show_filter_and_preview(self, actual_col: str = None):
        st.subheader("🔍 Review Specific Records")

        # options = ["All Predicted Records"]
        # if actual_col and actual_col in self.df.columns:
        #     options.append("False Negatives (Actual=0, Predicted=1)")

        # selected = st.radio("Show:", options=options, horizontal=True)

        # if selected == "False Negatives (Actual=0, Predicted=1)":
        #     filtered_df = self.df[
        #         (self.df[actual_col] == 0) & (self.df[self.prediction_col] == 1)
        #     ]
        # else:
        #     filtered_df = self.df

        st.dataframe(self.df.head(100), use_container_width=True)
