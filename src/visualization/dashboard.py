# src/visualization/dashboard.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, df: pd.DataFrame, prediction_col: str = 'prediction'):
        self.df = df
        self.prediction_col = prediction_col

    def show_model_metrics(self, metrics):
        """Enhanced model metrics display with beautiful cards"""
        st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        st.markdown(f"<h3>🧠 {metrics['Model']}</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create metric cards with better styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">📊 Accuracy</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{metrics['Accuracy']}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">🎯 F1 Score</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{metrics['F1 Score']}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">📈 ROC AUC</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{metrics['ROC AUC']}%</h2>
            </div>
            """, unsafe_allow_html=True)

    def show_before_after_churn_comparison(self, actual_col: str = None):
        """Enhanced churn comparison with better styling"""
        st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        st.markdown("<h3>📊 Before vs. After: Churn Comparison</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
                title="Before vs. After: Churn Distribution (%)",
                color_discrete_map={"Actual": "#3498db", "Predicted": "#e74c3c"}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#2c3e50'),
                title_font_color='#4886c4'
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_3d_distribution(self):
        """Enhanced 3D distribution with better styling"""
        st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        st.markdown("<h3>🧊 Churn Insights (3D Scatter Plot)</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Add a filter for prediction value
        selected = st.selectbox("Filter 3D plot by prediction", options=["All", "Churned", "Not Churned"])
        if selected == "Churned":
            data = self.df[self.df[self.prediction_col] == 1]
        elif selected == "Not Churned":
            data = self.df[self.df[self.prediction_col] == 0]
        else:
            data = self.df.head(50000)
            
        data = data.sample(n=min(len(data), 50000), random_state=42)

        numeric_cols = data.select_dtypes(include='number').drop(columns=[self.prediction_col], errors='ignore')

        if len(numeric_cols.columns) >= 3:
            fig = px.scatter_3d(
                data,
                x=numeric_cols.columns[0],
                y=numeric_cols.columns[1],
                z=numeric_cols.columns[2],
                color=self.prediction_col,
                title="3D Feature Scatter by Churn",
                color_discrete_map={0: "#3498db", 1: "#e74c3c"}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#2c3e50'),
                title_font_color="#4886c4"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for 3D plot.")

    def show_summary_cards(self):
        """Enhanced summary cards with better styling"""
        st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        st.markdown("<h3>📌 Summary Statistics</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        total = len(self.df)
        churned = int((self.df[self.prediction_col] == 1).sum())
        not_churned = total - churned
        churn_rate = (churned / total) * 100 if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">👥 Total Customers</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{total:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">❌ Churned</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{churned:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">✅ Not Churned</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{not_churned:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="color: #ecf0f1; font-weight: 500;">📊 Churn Rate</h4>
                <h2 style="color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{churn_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

    def show_filter_and_preview(self, actual_col: str = None):
        """Enhanced data preview with filtering options"""
        st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        st.markdown("<h3>🔍 Data Preview & Analysis</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Add filtering options
        filter_options = ["All Records", "Churned Only", "Not Churned Only"]
        if actual_col and actual_col in self.df.columns:
            filter_options.extend(["False Positives", "False Negatives", "True Positives", "True Negatives"])
        
        selected_filter = st.selectbox("Filter records:", filter_options)
        
        if selected_filter == "Churned Only":
            filtered_df = self.df[self.df[self.prediction_col] == 1]
        elif selected_filter == "Not Churned Only":
            filtered_df = self.df[self.df[self.prediction_col] == 0]
        elif selected_filter == "False Positives" and actual_col and actual_col in self.df.columns:
            filtered_df = self.df[(self.df[actual_col] == 0) & (self.df[self.prediction_col] == 1)]
        elif selected_filter == "False Negatives" and actual_col and actual_col in self.df.columns:
            filtered_df = self.df[(self.df[actual_col] == 1) & (self.df[self.prediction_col] == 0)]
        elif selected_filter == "True Positives" and actual_col and actual_col in self.df.columns:
            filtered_df = self.df[(self.df[actual_col] == 1) & (self.df[self.prediction_col] == 1)]
        elif selected_filter == "True Negatives" and actual_col and actual_col in self.df.columns:
            filtered_df = self.df[(self.df[actual_col] == 0) & (self.df[self.prediction_col] == 0)]
        else:
            filtered_df = self.df

        st.write(f"**Showing limited records**")
        st.dataframe(filtered_df.head(100), use_container_width=True)

    def show_confusion_matrix(self, actual_col: str = None):
        """Show confusion matrix if actual values are available"""
        if actual_col and actual_col in self.df.columns:
            st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
            st.markdown("<h3>📊 Confusion Matrix</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            cm = confusion_matrix(self.df[actual_col], self.df[self.prediction_col])
            
            # Create confusion matrix heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                showscale=True
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=500,
                height=400,
                font=dict(color='#2c3e50'),
                title_font_color='#4886c4'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show classification report
            report = classification_report(self.df[actual_col], self.df[self.prediction_col], output_dict=True)
            st.write("**Classification Report:**")
            st.dataframe(pd.DataFrame(report).transpose())

    def show_feature_importance(self, feature_columns=None):
        """Show feature importance if available"""
        if feature_columns and len(feature_columns) > 0:
            st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
            st.markdown("<h3>🎯 Feature Importance Analysis</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate correlation with target
            if self.prediction_col in self.df.columns:
                correlations = []
                for col in feature_columns:
                    if col in self.df.columns:
                        corr = abs(self.df[col].corr(self.df[self.prediction_col]))
                        correlations.append({"Feature": col, "Correlation": corr})
                
                if correlations:
                    corr_df = pd.DataFrame(correlations)
                    corr_df = corr_df.sort_values("Correlation", ascending=False)
                    
                    fig = px.bar(
                        corr_df.head(10),
                        x="Correlation",
                        y="Feature",
                        orientation='h',
                        title="Top 10 Feature Correlations with Target",
                        color="Correlation",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12, color='#2c3e50'),
                        title_font_color='#4886c4'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def show_prediction_distribution(self):
        """Show prediction distribution"""
        st.markdown('<div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.2rem; border-radius: 12px; color: white; text-align: center; margin: 1rem 0; box-shadow: 0 3px 10px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
        st.markdown("<h3>📈 Prediction Distribution</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if self.prediction_col in self.df.columns:
            fig = px.histogram(
                self.df,
                x=self.prediction_col,
                title="Distribution of Predictions",
                color_discrete_sequence=["#3498db"]
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#2c3e50'),
                title_font_color='#4886c4'
            )
            st.plotly_chart(fig, use_container_width=True)
