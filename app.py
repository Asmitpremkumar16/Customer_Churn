# IMPORT PACKAGES

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# SET PAGE CONFIG

st.set_page_config(page_title= "Churn Predictor", layout= 'wide')


# LOAD DATA

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
  
# LOADING MODEL AND PREPROCESSOR

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

# UI DESIGN

st.sidebar.title("Churn Model Analysis")

option= st.sidebar.radio("Select an Option", ("Analysis","Charts & Graphs","Churn Model","About"))

if option == "Analysis":
    # ── Raw Data ──────────────────────────────────────────────────────────────
    st.header("Raw DataFrame", divider=True)
    st.dataframe(data.head(5), use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.header("Overview", divider=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(data))
    with col2:
        st.metric("Churned", data["Exited"].sum())
    with col3:
        st.metric("Churn Rate", f"{data['Exited'].mean()*100:.1f}%")

    # ── Grouped Analysis ──────────────────────────────────────────────────────
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

if option == "Charts & Graphs":
    st.header("Charts", divider=True)

    # ── Chart 1 — Geography by Exited ────────────────────────────────────────
    fig1 = px.histogram(
        data, x="Geography",
        color="IsActiveMember",
        facet_col="Exited",
        barmode='group',
        title="Geography by Exited",
        text_auto=True,
        labels={"IsActiveMember": "Active Member"}  # ✅ readable legend
    )
    fig1.update_traces(textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2 — Correlation Heatmap ────────────────────────────────────────
    df = data[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
               "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]].corr()
    fig2 = px.imshow(df, text_auto=".2f", title="Correlation Heatmap",
                     color_continuous_scale="RdBu_r")  # ✅ better color scale
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3 & 4 side by side ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.histogram(
            data, x="Age",
            color="Exited",
            barmode='overlay',
            opacity=0.6,
            title="Age Distribution — Churned vs Retained",
            labels={"Exited": "Exited"}      # ✅ readable legend
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.histogram(
            data, x="EstimatedSalary",
            color="Exited",
            opacity=0.6,
            title="Churn across Estimated Salary",  # ✅ fixed title (was "Balance")
            labels={"Exited": "Exited"}
        )
        st.plotly_chart(fig4, use_container_width=True)
    col5= st.columns(1)

if option == "Churn Model":
   pass