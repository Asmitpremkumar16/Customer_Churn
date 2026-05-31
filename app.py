import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="🏦", layout="centered")

# ── Model ─────────────────────────────────────────────────────────────────────
class ChurnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(11, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):        
        return self.network(x)

# ── Load model & preprocessor ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = ChurnModel()
    model.load_state_dict(torch.load('ChurnModel.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_preprocessor():
    with open('preprocessor.pkl', 'rb') as f:
        return joblib.load(f)

model = load_model()
preprocessor = load_preprocessor()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Churn Predictor")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score",300, 850, 600)
    age          = st.number_input("Age", 18, 100, 35)
    tenure       = st.number_input("Tenure (years)", 0, 10, 5)
    balance      = st.number_input("Balance ($)", 0.0, 250000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    salary      = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active   = st.selectbox("Is Active Member", ["Yes", "No"])
    geography   = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender      = st.selectbox("Gender", ["Female", "Male"])

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Churn"):

    # Build raw dataframe — no manual encoding needed!
    input_df = pd.DataFrame([{
        'CreditScore':     credit_score,
        'Age':             age,
        'Tenure':          tenure,
        'Balance':         balance,
        'EstimatedSalary': salary,
        'Geography':       geography,
        'Gender':          gender,
        'NumOfProducts':   num_products,
        'HasCrCard':       1 if has_cr_card == "Yes" else 0,
        'IsActiveMember':  1 if is_active   == "Yes" else 0
    }])

    # Preprocessor handles all encoding and scaling automatically
    input_processed = preprocessor.transform(input_df)
    input_tensor = torch.tensor(input_processed, dtype=torch.float32)

    # Predict
    with torch.inference_mode():
        prob = model(input_tensor).item()
        prediction = 1 if prob >= 0.5 else 0

    # Result
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {prob*100:.1f}% probability")
    else:
        st.success(f"✅ Low Churn Risk — {(1-prob)*100:.1f}% probability to stay")

    st.progress(float(prob))
    st.caption(f"Raw probability: {prob:.4f}")