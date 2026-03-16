import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, DistanceMetric

from utils import grapher
from utils import data
from utils import features


def progressBar(delta: float):
    progress = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i+1)
        progress.text(f"{i+1}%")
        time.sleep(delta)


##############################
#   Data Inspection
##############################
st.title("Introduction")
df = data.initData("data/raw/patients.csv")
st.dataframe(df.head())

st.title("Descriptive Statistics")
left, right = st.columns(2)
with left:
    st.dataframe(df.describe())
with right:
    st.dataframe(data.summary(df))

##############################
#   PreProcess
##############################
st.divider()

if (st.button("Pre-Process")):
    plcText = st.empty()
    plcText.text(f"Removing Duplicates and Null-Rows")
    progressBar(0.05)
    dup, null = data.preprocess(df)
    plcText.text(f"Removed {dup} Duplicate Rows and {null} Null Rows")


st.divider()

##############################
#   EDA
##############################
st.title("Eploratory Data Analysis")

st.header("Featues Distribution")
grapher.pltfeaturesDistribution(df)

st.header("Features Box Plot")
grapher.pltboxDistribution(df)

st.divider()

##############################
#   Feature Selection
##############################
st.title("Feature Selection")

st.header("Correlation of Features")
method = (
    st.select_slider(
        "Select Correlation Method",
        options= ["pearson", "kendall", "spearman"])
)
df_encoded = data.encode(df, {"Low":0, "Medium":1, "High":2})
correlation = features.getCorrelation(df_encoded, method)
targetCorrelation = (
    correlation["Level"]
    .sort_values(ascending=False)
    .drop(index="Level")
    if correlation is not None else pd.Series()
    # Case: getCorr() -> None
)
st.dataframe(targetCorrelation)

thrsh = st.slider("Select Threshold for Features", min_value=25, max_value=100)
topk = targetCorrelation[ targetCorrelation >= thrsh ]
st.dataframe(topk)

st.divider()

##############################
#   Feature Scaling
##############################
st.title("Feature Scaling")
scaler = StandardScaler()
# Scalar discards column names
df[topk.index] = scaler.fit_transform(df[topk.index])
st.dataframe(df[topk.index])

##############################
#   Train/Test Split
##############################
st.title("Train/Test Split")

sizeTrain = st.slider("Select Training Size", min_value=0.5, max_value=0.90, step=0.01)
sizeTest = 1.0 - sizeTrain

x_train, x_test, y_train, y_test = train_test_split(
    df[topk.index],
    df["Level"],
    train_size=sizeTrain,
    test_size=sizeTest
)

st.divider()