import streamlit as st
import pandas as pd

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
