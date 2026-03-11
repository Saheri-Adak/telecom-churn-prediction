# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

#  Load Model 
@st.cache_resource
def load_model():
    return joblib.load(r'C:\Users\ADMIN\OneDrive\Desktop\Customer Churn Prediction\Models\xgb_model.pkl')

model = load_model()

#Header
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Enter customer details below to predict whether they are likely to churn.")
st.divider()

#Input Form 
st.subheader("👤 Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender           = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner          = st.selectbox("Partner", ["Yes", "No"])
    dependents       = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.markdown("**Services**")
    phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security  = st.selectbox("Online Security", ["Yes", "No"])
    online_backup    = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection= st.selectbox("Device Protection", ["Yes", "No"])
    tech_support     = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

with col3:
    st.markdown("**Account Info**")
    tenure           = st.slider("Tenure (Months)", 0, 72, 12)
    contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing= st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method   = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges    = st.number_input("Total Charges ($)",
                            min_value=0.0, max_value=10000.0,
                            value=float(tenure * monthly_charges))

st.divider()

# Predict Button
if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):

    # Build input dataframe
    input_data = pd.DataFrame([{
        'gender':            gender,
        'SeniorCitizen':     1 if senior_citizen == "Yes" else 0,
        'Partner':           partner,
        'Dependents':        dependents,
        'tenure':            tenure,
        'PhoneService':      phone_service,
        'MultipleLines':     multiple_lines,
        'InternetService':   internet_service,
        'OnlineSecurity':    online_security,
        'OnlineBackup':      online_backup,
        'DeviceProtection':  device_protection,
        'TechSupport':       tech_support,
        'StreamingTV':       streaming_tv,
        'StreamingMovies':   streaming_movies,
        'Contract':          contract,
        'PaperlessBilling':  paperless_billing,
        'PaymentMethod':     payment_method,
        'MonthlyCharges':    monthly_charges,
        'TotalCharges':      total_charges
    }])

    # Predict
    churn_prob  = model.predict_proba(input_data)[0][1]
    churn_label = "Yes" if churn_prob >= 0.5 else "No"

    # Result 
    st.subheader("🎯 Prediction Result")
    col1, col2 = st.columns(2)

    with col1:
        if churn_label == "Yes":
            st.error(f"⚠️ This customer is **likely to churn**")
        else:
            st.success(f"✅ This customer is **likely to stay**")

        st.metric("Churn Probability", f"{churn_prob:.1%}")
        st.progress(float(churn_prob))

    with col2:
        # Risk level
        if churn_prob >= 0.75:
            risk = "🔴 High Risk"
        elif churn_prob >= 0.5:
            risk = "🟠 Medium Risk"
        elif churn_prob >= 0.25:
            risk = "🟡 Low Risk"
        else:
            risk = "🟢 Very Low Risk"

        st.metric("Risk Level", risk)
        st.markdown(f"""
        **Recommended Action:**
        {"🚨 Immediate retention offer needed!" if churn_prob >= 0.75
         else "📞 Schedule a follow-up call." if churn_prob >= 0.5
         else "📧 Send a loyalty reward." if churn_prob >= 0.25
         else "😊 Customer is happy — no action needed."}
        """)

    #SHAP Explanation
    st.divider()
    st.subheader("🔍 Why This Prediction?")

    preprocessor        = model.named_steps['preprocessor']
    classifier          = model.named_steps['classifier']
    input_processed     = preprocessor.transform(input_data)
    feature_names_raw   = preprocessor.get_feature_names_out()
    feature_names_clean = [n.split('__')[-1] for n in feature_names_raw]
    input_df            = pd.DataFrame(input_processed, columns=feature_names_clean)

    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = input_df.iloc[0],
            feature_names = feature_names_clean
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()