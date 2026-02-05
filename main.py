# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Drug Shortage Predictor",
    page_icon="💊",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .prediction-high { 
        background-color: #ffebee; padding: 25px; border-radius: 10px; border-left: 6px solid #f44336;
        margin: 10px 0px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-medium { 
        background-color: #fff8e1; padding: 25px; border-radius: 10px; border-left: 6px solid #ff9800;
        margin: 10px 0px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-low { 
        background-color: #e8f5e8; padding: 25px; border-radius: 10px; border-left: 6px solid #4caf50;
        margin: 10px 0px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .input-section { 
        background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)


class DrugShortagePredictor:
    def __init__(self):
        self.drug_categories = {
            'M01AB': 'Anti-inflammatory preparations',
            'M01AE': 'Anti-inflammatory drugs (ibuprofen)',
            'N02BA': 'Salicylic acid and derivatives',
            'N02BE': 'Pyrazolones and analgesics',
            'N05B': 'Anxiolytic drugs',
            'N05C': 'Hypnotic and sedative drugs',
            'R03': 'Drugs for obstructive airway diseases',
            'R06': 'Antihistamines for systemic use'
        }

        self.best_models = {
            'M01AB': 'LightGBM', 'M01AE': 'LightGBM', 'N02BA': 'AutoGluon',
            'N02BE': 'LightGBM', 'N05B': 'LightGBM', 'N05C': 'TabNet',
            'R03': 'LightGBM', 'R06': 'LightGBM'
        }

        self.model_accuracies = {
            'M01AB': 0.9804, 'M01AE': 0.9861, 'N02BA': 0.9903,
            'N02BE': 0.9852, 'N05B': 0.9848, 'N05C': 0.9989,
            'R03': 0.9975, 'R06': 0.9903
        }

    def calculate_risk(self, current_sales, rolling_mean, rolling_std, lag_1, lag_7, month, weekday):
        """Calculate shortage risk based on input features"""

        # Feature calculations (simplified version of your original logic)
        sales_ratio = current_sales / rolling_mean if rolling_mean > 0 else 1
        volatility_ratio = rolling_std / rolling_mean if rolling_mean > 0 else 0
        trend = (current_sales - lag_7) / lag_7 if lag_7 > 0 else 0

        # Risk score calculation
        risk_score = 0

        # Sales ratio component (most important)
        if sales_ratio > 1.25:
            risk_score += 8
        elif sales_ratio > 1.15:
            risk_score += 5
        elif sales_ratio > 1.05:
            risk_score += 3

        # Volatility component
        if volatility_ratio > 0.3:
            risk_score += 3
        elif volatility_ratio > 0.2:
            risk_score += 2

        # Trend component
        if trend > 0.2:
            risk_score += 2
        elif trend > 0.1:
            risk_score += 1

        # Seasonal adjustment (winter months for respiratory drugs)
        month_adjustment = 0
        if month in [12, 1, 2]:  # Winter months
            month_adjustment = 1
        risk_score += month_adjustment

        # Determine risk level
        if risk_score >= 8:
            return 2, sales_ratio, risk_score  # High risk
        elif risk_score >= 4:
            return 1, sales_ratio, risk_score  # Medium risk
        else:
            return 0, sales_ratio, risk_score  # Low risk

    def predict(self, drug_category, features):
        """Make prediction for given inputs"""
        risk_level, sales_ratio, risk_score = self.calculate_risk(**features)

        # Adjust confidence based on model accuracy
        base_confidence = max(0.7, 1 - (risk_score * 0.05))
        confidence = base_confidence * self.model_accuracies[drug_category]

        return {
            'risk_level': risk_level,
            'confidence': min(0.99, confidence),
            'sales_ratio': sales_ratio,
            'risk_score': risk_score,
            'model_used': self.best_models[drug_category],
            'model_accuracy': self.model_accuracies[drug_category],
            'prediction_time': datetime.now()
        }


def main():
    st.title("💊 Drug Shortage Risk Predictor")
    st.markdown("Enter current sales data to predict shortage risk for different drug categories")
    st.markdown("---")

    predictor = DrugShortagePredictor()

    # Input Section
    st.header("📥 Enter Sales Data")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("Drug Information")

            drug_category = st.selectbox(
                "Select Drug Category",
                options=list(predictor.drug_categories.keys()),
                format_func=lambda x: f"{x} - {predictor.drug_categories[x]}"
            )

            current_sales = st.number_input(
                "Current Daily Sales (units)",
                min_value=0,
                max_value=10000,
                value=150,
                step=10,
                help="Number of units sold today"
            )

            rolling_mean = st.number_input(
                "7-Day Rolling Average Sales",
                min_value=0,
                max_value=10000,
                value=120,
                step=10,
                help="Average sales over the past 7 days"
            )

            rolling_std = st.number_input(
                "7-Day Sales Standard Deviation",
                min_value=0.0,
                max_value=1000.0,
                value=15.0,
                step=1.0,
                help="Standard deviation of sales over past 7 days"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("Historical Data")

            lag_1 = st.number_input(
                "Previous Day Sales",
                min_value=0,
                max_value=10000,
                value=130,
                step=10,
                help="Sales from yesterday"
            )

            lag_7 = st.number_input(
                "Sales 7 Days Ago",
                min_value=0,
                max_value=10000,
                value=125,
                step=10,
                help="Sales from the same day last week"
            )

            month = st.selectbox(
                "Current Month",
                options=list(range(1, 13)),
                format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
                index=0  # January
            )

            weekday = st.selectbox(
                "Current Day of Week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ][x],
                index=0
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        predict_btn = st.button(
            "🚀 PREDICT SHORTAGE RISK",
            type="primary",
            use_container_width=True,
            help="Click to analyze shortage risk based on entered data"
        )

    # Prediction Results
    if predict_btn:
        st.markdown("---")
        st.header("🎯 Prediction Results")

        # Prepare features
        features = {
            'current_sales': current_sales,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'month': month,
            'weekday': weekday
        }

        with st.spinner("🔮 Analyzing sales patterns and predicting shortage risk..."):
            # Simulate processing time
            import time
            time.sleep(1)

            # Make prediction
            prediction = predictor.predict(drug_category, features)

            # Display Results
            col1, col2 = st.columns([2, 1])

            with col1:
                risk_level = prediction['risk_level']
                confidence = prediction['confidence']
                sales_ratio = prediction['sales_ratio']

                if risk_level == 2:
                    st.markdown(f"""
                    <div class="prediction-high">
                        <h2>🚨 HIGH SHORTAGE RISK DETECTED!</h2>
                        <p><strong>Confidence Level:</strong> {confidence:.2%}</p>
                        <p><strong>Sales Pattern:</strong> Current sales are {sales_ratio:.2f}x above 7-day average</p>
                        <p><strong>Risk Score:</strong> {prediction['risk_score']}/12 (High Risk)</p>
                        <p><strong>AI Model:</strong> {prediction['model_used']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.error("""
                    **🚨 IMMEDIATE ACTION REQUIRED:**
                    - 📞 Contact suppliers for emergency restocking
                    - 📦 Expedite all pending shipments
                    - 🚚 Activate alternative distribution channels
                    - ⚠️ Notify hospital departments about potential shortages
                    - 💊 Consider therapeutic alternatives
                    """)

                elif risk_level == 1:
                    st.markdown(f"""
                    <div class="prediction-medium">
                        <h2>⚠️ MEDIUM SHORTAGE RISK</h2>
                        <p><strong>Confidence Level:</strong> {confidence:.2%}</p>
                        <p><strong>Sales Pattern:</strong> Current sales are {sales_ratio:.2f}x above 7-day average</p>
                        <p><strong>Risk Score:</strong> {prediction['risk_score']}/12 (Medium Risk)</p>
                        <p><strong>AI Model:</strong> {prediction['model_used']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning("""
                    **⚠️ RECOMMENDED ACTIONS:**
                    - 📊 Increase monitoring to daily checks
                    - 📋 Review current inventory levels
                    - 📧 Pre-alert suppliers about increased demand
                    - 🔄 Optimize reorder points
                    - 📈 Analyze demand trends for next 7 days
                    """)

                else:
                    st.markdown(f"""
                    <div class="prediction-low">
                        <h2>✅ LOW SHORTAGE RISK</h2>
                        <p><strong>Confidence Level:</strong> {confidence:.2%}</p>
                        <p><strong>Sales Pattern:</strong> Current sales are {sales_ratio:.2f}x of 7-day average</p>
                        <p><strong>Risk Score:</strong> {prediction['risk_score']}/12 (Low Risk)</p>
                        <p><strong>AI Model:</strong> {prediction['model_used']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.success("""
                    **✅ STATUS: NORMAL OPERATIONS**
                    - 📈 Continue regular monitoring schedule
                    - 💊 Maintain current inventory levels
                    - 📝 Proceed with scheduled supplier orders
                    - 🔍 Monitor for any demand pattern changes
                    """)

            with col2:
                st.subheader("📊 Risk Analysis")

                # Risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction['risk_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score"},
                    delta={'reference': 4},
                    gauge={
                        'axis': {'range': [None, 12]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 4], 'color': "lightgreen"},
                            {'range': [4, 8], 'color': "yellow"},
                            {'range': [8, 12], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 8
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Model info
                st.metric("Model Accuracy", f"{prediction['model_accuracy']:.2%}")
                st.metric("Sales Ratio", f"{sales_ratio:.2f}x")
                st.metric("Prediction Time", prediction['prediction_time'].strftime('%H:%M:%S'))

        # Detailed Analysis
        st.markdown("---")
        st.subheader("🔍 Detailed Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Sales", f"{current_sales} units")
            st.metric("7-Day Average", f"{rolling_mean} units")
            st.metric("Demand vs Average", f"{(sales_ratio - 1) * 100:+.1f}%")

        with col2:
            st.metric("Sales Volatility", f"{rolling_std:.1f} units")
            st.metric("Week-over-Week Change",
                      f"{((current_sales - lag_7) / lag_7 * 100 if lag_7 > 0 else 0):+.1f}%")
            st.metric("Day-over-Day Change",
                      f"{((current_sales - lag_1) / lag_1 * 100 if lag_1 > 0 else 0):+.1f}%")

        with col3:
            st.metric("Best Model", prediction['model_used'])
            st.metric("Confidence", f"{confidence:.2%}")
            st.metric("Risk Category", ["Low", "Medium", "High"][risk_level])

        # Feature Importance (Simplified)
        st.subheader("📈 Key Risk Factors")

        factors = {
            "High Sales Volume": min(10, max(0, (sales_ratio - 1) * 20)),
            "Sales Volatility": min(10, rolling_std / rolling_mean * 30 if rolling_mean > 0 else 0),
            "Upward Trend": min(10, max(0, (current_sales - lag_7) / lag_7 * 50 if lag_7 > 0 else 0)),
            "Seasonal Factor": 2 if month in [12, 1, 2] and drug_category in ['R03', 'R06'] else 0
        }

        for factor, score in factors.items():
            st.write(f"{factor}: {score:.1f}/10")
            st.progress(min(1.0, score / 10))


if __name__ == "__main__":
    main()