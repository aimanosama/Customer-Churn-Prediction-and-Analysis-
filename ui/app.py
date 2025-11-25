import pandas as pd
import streamlit as st
import joblib

used = pd.read_csv("data/processed/X_train_scaled.csv")
input_ = pd.read_csv("data/processed/churn_cleaned.csv")

model = joblib.load("models/best.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

states = input_["State"].unique()

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📞 Customer Churn Predictor")
st.write("This app predicts if a customer will churn or not.")
st.sidebar.header("Customer features")

state = st.sidebar.selectbox("State", states)
account_length = st.sidebar.number_input("Account length (days)", 1, 300, 1)
international_plan = st.sidebar.number_input("International plan", 0,1,0)
voice_mail_plan = st.sidebar.number_input("Voice mail plan", 0,1,0)
number_vmail_messages = st.sidebar.number_input("Number of vmail messages", 0, 100, 0)

total_day_minutes = st.sidebar.number_input("Total day minutes", 0.0, 1000.0, 0.0)
total_day_calls = st.sidebar.number_input("Total day calls", 0, 200, 0)
total_day_charge = st.sidebar.number_input("Total day charge", 0.0, 100.0, 0.0)

total_eve_minutes = st.sidebar.number_input("Total evening minutes", 0.0, 1000.0, 0.0)
total_eve_calls = st.sidebar.number_input("Total evening calls", 0, 200, 0)
total_eve_charge = st.sidebar.number_input("Total evening charge", 0.0, 100.0, 0.0)

total_night_minutes = st.sidebar.number_input("Total night minutes", 0.0, 1000.0, 0.0)
total_night_calls = st.sidebar.number_input("Total night calls", 0, 200, 0)
total_night_charge = st.sidebar.number_input("Total night charge", 0.0, 100.0, 0.0)

total_intl_minutes = st.sidebar.number_input("Total international minutes", 0.0, 100.0, 0.0)
total_intl_calls = st.sidebar.number_input("Total international calls", 0, 30, 0)
total_intl_charge = st.sidebar.number_input("Total international charge", 0.0, 20.0,0.0)

customer_service_calls = st.sidebar.number_input("Customer service calls", 0, 20, 0)

total_national_minutes = total_day_minutes + total_eve_minutes + total_night_minutes
total_national_calls = total_day_calls + total_eve_calls + total_night_calls
total_national_charge = total_day_charge + total_eve_charge + total_night_charge

avg_minutes_per_call = (
    total_national_minutes / total_national_calls if total_national_calls > 0 else 0
)
avg_int_minutes_per_call = (
    total_intl_minutes / total_intl_calls if total_intl_calls > 0 else 0
)
cost_per_minute = (
    total_national_charge / total_national_minutes if total_national_minutes > 0 else 0
)
cost_per_minute_intl = (
    total_intl_charge / total_intl_minutes if total_intl_minutes > 0 else 0
)

high_service_calls = 1 if customer_service_calls > 3 else 0
tenure_category = (
    "Low" if account_length <= 74 else "Medium" if account_length <= 127 else "High"
)
has_all_plans = 1 if (international_plan == 1 and voice_mail_plan == 1) else 0
zero_vmail_messages = 1 if number_vmail_messages == 0 else 0

# Create DataFrame for prediction
data = pd.DataFrame([{
    "State": state,
    "Tenure category": tenure_category,
    "International plan": international_plan,
    "Voice mail plan": voice_mail_plan,
    "Total day minutes": total_day_minutes,
    "Total day charge": total_day_charge,
    "Total eve minutes": total_eve_minutes,
    "Total eve charge": total_eve_charge,
    "Total night minutes": total_night_minutes,
    "Total night charge": total_night_charge,
    "Total intl minutes": total_intl_minutes,
    "Total intl calls": total_intl_calls,
    "Total intl charge": total_intl_charge,
    "Customer service calls": customer_service_calls,
    "Total national minutes": total_national_minutes,
    "Total national calls": total_national_calls,
    "Total national charge": total_national_charge,
    "Avg minutes per call": avg_minutes_per_call,
    "Avg int minutes per call": avg_int_minutes_per_call,
    "Cost per minute": cost_per_minute,
    "Cost per minute intl": cost_per_minute_intl,
    "High service calls": high_service_calls,
    "Has All Plans": has_all_plans,
    "zero_vmail_messages": zero_vmail_messages
}])

if st.button("Predict Churn"):
    X = preprocessor.transform(data)
    pred = model.predict(X)[0]

    churn_text = "🚨 Likely to Churn" if pred == 1 else "✅ Not Likely to Churn"
    st.subheader(f"Prediction: {churn_text}")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
        st.write(f"Churn Probability: **{prob:.2f}**")

st.markdown("---")
