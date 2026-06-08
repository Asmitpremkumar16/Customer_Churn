# IMPORT PACKAGES

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, confusion_matrix


# SET PAGE CONFIG

st.set_page_config(page_title= "Churn Predictor", layout= 'wide')


# LOAD DATA----------------------------------------------------------------------------------

data= pd.read_csv("churn modelling.csv")

# MODEL 

class ChurnModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.network= nn.Sequential(
        nn.Linear(in_features= 11, out_features= 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features= 64, out_features= 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features= 128, out_features= 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features= 64, out_features= 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features=32, out_features= 10),
        nn.ReLU(),
        nn.Linear(in_features=10, out_features= 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.network(x)
  
# LOADING MODEL AND PREPROCESSOR----------------------------------------------------------------------------------

@st.cache_resource
def load_model():
  model= ChurnModel()
  model.load_state_dict(torch.load(f= "ChurnModel.pth", map_location= torch.device('cpu')))
  model.eval()
  return model

@st.cache_resource
def load_preprocessor():
  with open("preprocessor.pkl", "rb") as f:
    return joblib.load(f)

model= load_model()
preprocessor= load_preprocessor()

# UI DESIGN----------------------------------------------------------------------------------

st.sidebar.title("Churn Model Analysis")

option= st.sidebar.radio("Select an Option", ("Analysis","Charts & Graphs","Churn Model","Model Performance","About"))

# Analysis Section----------------------------------------------------------------------------------

if option == "Analysis":
    # Raw Data
    st.header("Raw DataFrame", divider=True)
    st.dataframe(data.head(5), use_container_width=True)

    #  Metrics 
    st.header("Overview", divider=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(data))
    with col2:
        st.metric("Churned", data["Exited"].sum())
    with col3:
        st.metric("Churn Rate", f"{data['Exited'].mean()*100:.1f}%")

    #  Grouped Analysis using groupby
    st.header("Churn Analysis", divider=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Exit Rate Overall (%)**")
        st.dataframe(
            data["Exited"]
            .value_counts(normalize=True)
            .mul(100).round(2)
            .rename(index={0: "Stayed", 1: "Exited"})
            .reset_index()
            .rename(columns={"index": "Category", "Exited": "Percentage (%)"}),
            use_container_width=True
        )

    with col2:
        st.markdown("**Exit Rate by Gender & Geography (%)**")
        st.dataframe(
            data.groupby(["Gender", "Geography"])["Exited"]
            .mean().mul(100).round(2)
            .reset_index()
            .rename(columns={"Exited": "Exit Rate (%)"}),
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Exit Rate by Gender & Active Member (%)**")
        st.dataframe(
            data.groupby(["Gender", "IsActiveMember"])["Exited"]
            .mean().mul(100).round(2)
            .reset_index()
            .rename(columns={
                "IsActiveMember": "Active Member",
                "Exited": "Exit Rate (%)"
            }),
            use_container_width=True
        )

    with col4:
        st.markdown("**Exit Rate by Products & Active Member (%)**")
        st.dataframe(
            data.groupby(["NumOfProducts", "IsActiveMember"])["Exited"]
            .mean().mul(100).round(2)
            .reset_index()
            .rename(columns={
                "NumOfProducts":  "No. of Products",
                "IsActiveMember": "Active Member",
                "Exited":         "Exit Rate (%)"
            }),
            use_container_width=True
        )
    st.info("""
    **Key Insights from Analysis:**
    - **Geography** - Germany has the highest churn rate compared to France and Spain
    - **Gender** - Female customers churn more than Male customers
    - **Active Members** - Inactive members are significantly more likely to churn
    """)

# Charts and Graphs Section----------------------------------------------------------------------------------

if option == "Charts & Graphs":
    st.header("Charts", divider=True)

    #  Geography by Exited 
    fig1 = px.histogram(
        data, x="Geography",
        color="IsActiveMember",
        facet_col="Exited",
        barmode='group',
        title="Geography by Exited",
        text_auto=True,
        labels={"IsActiveMember": "Active Member"}
    )
    fig1.update_traces(textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

    #  Correlation Heatmap 
    df = data[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
               "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]].corr()
    fig2 = px.imshow(df, text_auto=".2f", title="Correlation Heatmap",
                     color_continuous_scale="RdBu_r")
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)

    #  Heatmap Info
    st.info("""
    **Key Insights from Correlation Heatmap:**
    - **Age** - Older customers are more likely to churn (as below histogram plot)
    - **Balance** - Higher balance customers tend to churn slightly more
    - **IsActiveMember** - Inactive members show the strongest negative correlation with retention
    """)

    #  Chart 3 & 4 side by side 
    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.histogram(
            data, x="Age",
            color="Exited",
            barmode='overlay',
            opacity=0.6,
            title="Age Distribution — Churned vs Retained",
            labels={"Exited": "Exited"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.histogram(
            data, x="EstimatedSalary",
            color="Exited",
            opacity=0.6,
            title="Churn across Estimated Salary",
            labels={"Exited": "Exited"}
        )
        st.plotly_chart(fig4, use_container_width=True)
    col5= st.columns(1)


# Churn Prediction Section----------------------------------------------------------------------------------

if option == "Churn Model":
    st.header("Churn Model", divider= True)
    
    # Columns Creation for UI with default inputs 
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
    
    # Prediction Button
    if st.button("Predict Churn"):

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

        #  Processing the input dataframe

        input_processed = preprocessor.transform(input_df)
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)

        with torch.inference_mode():
            prob = model(input_tensor).item()
            prediction = 1 if prob >= 0.5 else 0

        if prediction == 1:
            st.error(f"High Churn Risk — {prob*100:.1f}% probability")
        else:
            st.success(f"Low Churn Risk — {(1-prob)*100:.1f}% probability to stay")

        st.progress(float(prob))
        st.caption(f"Raw probability of leaving: {prob:.4f}") 


# Model Performance Section----------------------------------------------------------------------------------


if option == "Model Performance":
    st.header("Model Performance", divider=True)
 
    # Results from the ANN model
    report_data = {
        "Class":      ["Stayed (0)", "Churned (1)", "Macro Avg", "Weighted Avg"],
        "Precision":  [0.93,         0.47,          0.70,        0.83],
        "Recall":     [0.79,         0.75,          0.77,        0.78],
        "F1-Score":   [0.85,         0.58,          0.72,        0.80],
        "Support":    [1593,         407,           2000,        2000]
    }
 
    #  Top Metrics 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "78%")
    with col2:
        st.metric("Class 1 Recall", "75%", help="% of actual churners correctly identified")
    with col3:
        st.metric("Class 1 Precision", "47%", help="% of predicted churners that actually churned")
    with col4:
        st.metric("Class 1 F1 Score", "0.58", help="Balance between precision and recall")
 
    #  Classification Report Table 
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report_data), use_container_width=True)
 
    #  Confusion Matrix 
    st.subheader("Confusion Matrix")
 
    cm = [[1258, 335],   # Actual 0: Stayed
          [102,  305]]   # Actual 1: Churned
 
    fig_cm = ff.create_annotated_heatmap(
        cm,
        x=["Predicted Stayed", "Predicted Churned"],
        y=["Actual Stayed", "Actual Churned"],
        colorscale="Blues",
        showscale=True
    )
    fig_cm.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
 
    #  Precision Recall F1 Bar Chart 
    st.subheader("Precision vs Recall vs F1-Score")
 
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"] * 2,
        "Value":  [0.93, 0.79, 0.85, 0.47, 0.75, 0.58],
        "Class":  ["Stayed (0)"] * 3 + ["Churned (1)"] * 3
    })
 
    fig_bar = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        color="Class",
        barmode="group",
        title="Precision, Recall & F1-Score by Class",
        text_auto=".2f",
        range_y=[0, 1]
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)
 
    #  Model Info 
    st.subheader("Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame({
            "Parameter": ["Model Type", "Input Features", "Hidden Layers", "Optimizer", "Loss Function", "Early Stopping"],
            "Value": ["ANN", "11", "4", "Adam", "BCELoss", "Yes (patience=50)"]
        }), use_container_width=True)
    with col2:
        st.dataframe(pd.DataFrame({
            "Parameter": ["Learning Rate", "Dropout", "Batch Norm", "Epochs", "SMOTE", "Scaler"],
            "Value": ["0.01", "0.3", "Yes", "~83", "Yes", "StandardScaler"]
        }), use_container_width=True)


# About Section----------------------------------------------------------------------------------

if option == "About":
    st.header("About", divider=True)

    #  Project Overview 
    st.subheader("Bank Customer Churn Prediction")
    st.markdown("""
    This app predicts whether a bank customer is likely to **churn (leave)** 
    using an Artificial Neural Network trained on real customer data.
    Built with **PyTorch** and deployed using **Streamlit**.
    """)

    #  Dataset Info 
    st.subheader("Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", "10,000")
    with col2:
        st.metric("Features", "11")
    with col3:
        st.metric("Churn Rate", "20%")

    st.markdown("""
    The dataset contains bank customer information including:
    - **Demographics** — Age, Gender, Geography
    - **Banking Info** — Balance, Credit Score, Tenure, Number of Products
    - **Behaviour** — Active Member status, Credit Card ownership
    """)

    #  Model Results 
    st.subheader("Model Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "78%")
    with col2:
        st.metric("Class 1 Recall", "75%")
    with col3:
        st.metric("Class 1 Precision", "47%")
    with col4:
        st.metric("Class 1 F1 Score", "0.58")

    #  Tech Stack 
    st.subheader("Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**PyTorch**\nModel Training")
    with col2:
        st.info("**Scikit-learn**\nPreprocessing")
    with col3:
        st.info("**Streamlit**\nDeployment")
    with col4:
        st.info("**Plotly**\nVisualizations")

    #  Developer 
    st.subheader("Developer")
    st.markdown("""
    **Asmit Prem Kumar**  
    [GitHub](https://github.com/Asmitpremkumar16/Customer_Churn)
    """)
    
