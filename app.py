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

@st.cache_data
def initData(path:str):
    df = (
        pd.read_csv(path, index_col="Patient Id")
        .drop(columns="index")
        .sort_index(ascending=True)
    )
    return df

def summary(df: pd.DataFrame):
    summary = pd.DataFrame({
        "Column":df.columns,
        "Null-Count":df.isnull().sum(),
    })
    summary.set_index("Column", inplace=True)
    return summary

def preprocess(df: pd.DataFrame):
    info = (df.duplicated().sum(), df.isnull().sum().sum())
    df.drop_duplicates(inplace=True)
    df.fillna("Unnown", inplace=True)
    return info

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
df = initData("data/raw/patients.csv")
st.dataframe(df.head())


st.title("Descriptive Statistics")
left, right = st.columns(2)
with left:
    st.dataframe(df.describe())
with right:
    st.dataframe(summary(df))
