import streamlit as st
import numpy as np
import pandas as pd
import string
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =====================
# FUNCTIONS
# =====================
def random_customer_id():
    num = ''.join(random.choices(string.digits, k=4))
    letters = ''.join(random.choices(string.ascii_uppercase, k=4))
    return f"{num}-{letters}"

@st.cache_data
def generate_data(n=500):
    np.random.seed(42)
    
    gender = np.random.choice(["Male", "Female"], size=n)
    senior = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = np.where(np.random.rand(n) < 0.55, "Yes", "No")
    dependents = np.where(np.random.rand(n) < 0.3, "Yes", "No")
    tenure = np.random.randint(0, 73, size=n)

    phone_service = np.where(np.random.rand(n) < 0.9, "Yes", "No")
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
    multiple_lines = np.array(multiple_lines)

    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.5, 0.15]
    )

    def dep_service(base_prob_yes, none_label="No internet service"):
        out = []
        for s in internet_service:
            if s == "No":
                out.append(none_label)
            else:
                out.append(np.random.choice(["Yes", "No"], p=[base_prob_yes, 1-base_prob_yes]))
        return np.array(out)

    online_security   = dep_service(0.36)
    online_backup     = dep_service(0.45)
    device_protection = dep_service(0.46)
    tech_support      = dep_service(0.34)
    streaming_tv      = dep_service(0.5)
    streaming_movies  = dep_service(0.5)

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.57, 0.22, 0.21]
    )
    paperless = np.where(
        (contract == "Month-to-month") & (np.random.rand(n) < 0.75), "Yes",
        np.where(np.random.rand(n) < 0.45, "Yes", "No")
    )
    payment_method = np.where(
        paperless == "Yes",
        np.random.choice(
            ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
            size=n, p=[0.55, 0.23, 0.22]
        ),
        np.random.choice(
            ["Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            size=n, p=[0.55, 0.25, 0.20]
        )
    )

    base_charge = np.where(
        internet_service == "No", np.random.uniform(15, 30, size=n),
        np.where(internet_service == "DSL", np.random.uniform(45, 75, size=n),
                 np.random.uniform(70, 110, size=n))
    )

    def add_if_yes(arr, amount):
        return np.where(arr == "Yes", amount, 0.0)

    monthly = (
        base_charge
        + add_if_yes(multiple_lines, 5)
        + add_if_yes(online_security, 6)
        + add_if_yes(online_backup, 5)
        + add_if_yes(device_protection, 5)
        + add_if_yes(tech_support, 7)
        + add_if_yes(streaming_tv, 8)
        + add_if_yes(streaming_movies, 8)
    )

    monthly = monthly + np.random.normal(0, 2.5, size=n)
    monthly = np.clip(monthly, 18, None).round(2)

    total = (monthly * tenure + np.random.normal(0, 20, size=n)).clip(min=0)
    total = np.where(tenure == 0, 0, total).round(2)

    logit = (
        -0.8
        + 0.9 * (contract == "Month-to-month").astype(float)
        - 0.6 * (contract == "Two year").astype(float)
        + 0.5 * (payment_method == "Electronic check").astype(float)
        + 0.4 * (internet_service == "Fiber optic").astype(float)
        - 0.5 * (online_security == "Yes").astype(float)
        - 0.5 * (tech_support == "Yes").astype(float)
        - 0.02 * tenure
        + 0.015 * (monthly - monthly.mean())
        + 0.25 * senior
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = np.where(np.random.rand(n) < prob_churn, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [random_customer_id() for _ in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly.astype(float),
        "TotalCharges": total.astype(float),
        "Churn": churn
    })
    
    return df

# =====================
# STREAMLIT APP
# =====================
st.set_page_config(page_title="Customer Churn Explorer", layout="wide")
st.title("📊 Customer Churn Prediction - Synthetic Data Explorer")

df = generate_data(500)

# =====================
# TRAIN SIMPLE MODEL
# =====================
X = df.drop(columns=["customerID","Churn"])
y = df["Churn"]

# Encode categorical
X_encoded = X.copy()
label_encoders = {}
for col in X_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =====================
# PREDICTION FORM
# =====================
st.sidebar.header("🔍 Predict Churn for a Customer")

gender = st.sidebar.selectbox("Gender", ["Male","Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0,1])
partner = st.sidebar.selectbox("Partner", ["Yes","No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes","No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes","No"])
multiple = st.sidebar.selectbox("Multiple Lines", ["Yes","No","No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes","No","No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes","No","No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes","No","No internet service"])
stream_tv = st.sidebar.selectbox("Streaming TV", ["Yes","No","No internet service"])
stream_movies = st.sidebar.selectbox("Streaming Movies", ["Yes","No","No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month","One year","Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes","No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
monthly = st.sidebar.slider("Monthly Charges", 18, 120, 50)
total = monthly * tenure

# Create DataFrame for input
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": stream_tv,
    "StreamingMovies": stream_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# Encode input
for col in input_data.select_dtypes(include="object").columns:
    le = label_encoders[col]
    input_data[col] = le.transform(input_data[col])

# Predict
pred = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

st.sidebar.subheader("Prediction Result")
if pred == "Yes":
    st.sidebar.error(f"❌ This customer is likely to CHURN (Fraud). Probability: {prob:.2f}")
else:
    st.sidebar.success(f"✅ This customer is likely to STAY (True). Probability: {prob:.2f}")

# =====================
# VISUALIZATIONS
# =====================
st.subheader("Churn Distribution")
fig, ax = plt.subplots(figsize=(6,4))
df["Churn"].value_counts().plot(kind="bar", color=["skyblue","salmon"], ax=ax)
ax.set_title("Churn Distribution")
st.pyplot(fig)

st.subheader("Tenure vs Churn")
fig, ax = plt.subplots(figsize=(8,5))
df[df["Churn"]=="Yes"]["tenure"].plot(kind="hist", bins=20, alpha=0.6, label="Churn=Yes", ax=ax)
df[df["Churn"]=="No"]["tenure"].plot(kind="hist", bins=20, alpha=0.6, label="Churn=No", ax=ax)
ax.legend()
ax.set_title("Tenure Distribution by Churn")
st.pyplot(fig)

st.subheader("Monthly Charges vs Churn")
fig, ax = plt.subplots(figsize=(8,5))
df[df["Churn"]=="Yes"]["MonthlyCharges"].plot(kind="hist", bins=20, alpha=0.6, label="Churn=Yes", ax=ax)
df[df["Churn"]=="No"]["MonthlyCharges"].plot(kind="hist", bins=20, alpha=0.6, label="Churn=No", ax=ax)
ax.legend()
ax.set_title("Monthly Charges Distribution by Churn")
st.pyplot(fig)

st.subheader("Contract Type vs Churn")
fig, ax = plt.subplots(figsize=(7,5))
pd.crosstab(df["Contract"], df["Churn"]).plot(kind="bar", stacked=True, color=["lightgreen","coral"], ax=ax)
ax.set_title("Contract Type vs Churn")
st.pyplot(fig)

