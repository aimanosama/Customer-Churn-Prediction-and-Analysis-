from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load(r"models/best.joblib")
preprocessor = joblib.load(r"models/preprocessor.joblib")

app = FastAPI(title="Customer Churn Predictor API")

# Define input schema
class CustomerData(BaseModel):
    State: str
    account_length: int
    international_plan: int
    voice_mail_plan: int
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    customer_service_calls: int


@app.post("/predict")
def predict_churn(data: CustomerData):
    # Derived features
    total_national_minutes = (
        data.total_day_minutes + data.total_eve_minutes + data.total_night_minutes
    )
    total_national_calls = (
        data.total_day_calls + data.total_eve_calls + data.total_night_calls
    )
    total_national_charge = (
        data.total_day_charge + data.total_eve_charge + data.total_night_charge
    )

    avg_minutes_per_call = (
        total_national_minutes / total_national_calls
        if total_national_calls > 0
        else 0
    )
    avg_int_minutes_per_call = (
        data.total_intl_minutes / data.total_intl_calls
        if data.total_intl_calls > 0
        else 0
    )
    cost_per_minute = (
        total_national_charge / total_national_minutes
        if total_national_minutes > 0
        else 0
    )
    cost_per_minute_intl = (
        data.total_intl_charge / data.total_intl_minutes
        if data.total_intl_minutes > 0
        else 0
    )

    high_service_calls = 1 if data.customer_service_calls > 3 else 0
    tenure_category = (
        "Low"
        if data.account_length <= 74
        else "Medium"
        if data.account_length <= 127
        else "High"
    )
    has_all_plans = 1 if (data.international_plan == 1 and data.voice_mail_plan == 1) else 0
    zero_vmail_messages = 1 if data.number_vmail_messages == 0 else 0

    df = pd.DataFrame(
        [
            {
                "State": data.State,
                "Tenure category": tenure_category,
                "International plan": data.international_plan,
                "Voice mail plan": data.voice_mail_plan,
                "Total day minutes": data.total_day_minutes,
                "Total day charge": data.total_day_charge,
                "Total eve minutes": data.total_eve_minutes,
                "Total eve charge": data.total_eve_charge,
                "Total night minutes": data.total_night_minutes,
                "Total night charge": data.total_night_charge,
                "Total intl minutes": data.total_intl_minutes,
                "Total intl calls": data.total_intl_calls,
                "Total intl charge": data.total_intl_charge,
                "Customer service calls": data.customer_service_calls,
                "Total national minutes": total_national_minutes,
                "Total national calls": total_national_calls,
                "Total national charge": total_national_charge,
                "Avg minutes per call": avg_minutes_per_call,
                "Avg int minutes per call": avg_int_minutes_per_call,
                "Cost per minute": cost_per_minute,
                "Cost per minute intl": cost_per_minute_intl,
                "High service calls": high_service_calls,
                "Has All Plans": has_all_plans,
                "zero_vmail_messages": zero_vmail_messages,
            }
        ]
    )

    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    churn_prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

    return {
        "prediction": int(pred),
        "prediction_text": "Likely to Churn" if pred == 1 else "Not Likely to Churn",
        "churn_probability": round(float(churn_prob), 2) if churn_prob else None,
    }


@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Predictor API"}
