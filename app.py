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

def pltHistogram(ax, df, feature):
    sns.histplot(
        data=df,
        x=feature,
        bins=df[feature].max() - df[feature].min(),
        kde=True,
        ax=ax
    )
    ax.set_title(
        f"{feature} Distribution",
        fontsize=12,
        fontweight="bold"
    )
    ax.axvline(
        df[feature].mean(),
        color='red',
        linestyle='--',
        label=f"Mean: {df[feature].mean():.1f}"
    )

    ax.legend()

def pltCountplot(ax, df, feature, xticks=None):
    sns.countplot(
        data=df,
        x="Gender",
        ax=ax
    )
    ax.set_title(
        f"{feature} Distribution",
        fontsize=12,
        fontweight="bold"
    )
    if xticks:
        ax.set_xticks([i for i in range(len(xticks))], xticks)

def pltBox(ax, df, feature):
    sns.boxplot(
        data=df,
        x=feature,
        ax=ax
    )
    ax.set_title(
        f"{feature} Distribution",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel(
        f"{"Age" if feature=="Age" else "Rating"}"
    )

@st.cache_data
def pltfeaturesDistribution():
    numeric_cols = [
        col for col in 
        df.select_dtypes(include="number").columns
        if col != "Gender"
    ]
    
    fig, axes = plt.subplots(5, 5, figsize=(30,15))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_cols):
        pltHistogram(axes[i], df, feature)

    pltCountplot(axes[22], df, "Gender", ["Male", "Female"])
    
    pltCountplot(axes[23], df, "Level")
    
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def pltboxDistribution():
    fig, axes = plt.subplots(5,5, figsize=(20,10))
    axes = axes.flatten()

    for i, feature in enumerate(df.columns):
        pltBox(axes[i], df, feature)

    plt.tight_layout()
    st.pyplot(fig)

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

##############################
#   PreProcess
##############################
st.divider()

if (st.button("Pre-Process")):
    plcText = st.empty()
    plcText.text(f"Removing Duplicates and Null-Rows")
    progressBar(0.05)
    dup, null = preprocess(df)
    plcText.text(f"Removed {dup} Duplicate Rows and {null} Null Rows")


st.divider()

##############################
#   EDA
##############################
st.title("Eploratory Data Analysis")

st.header("Featues Distribution")
pltfeaturesDistribution()

st.header("Features Box Plot")
pltboxDistribution()