import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/patients.csv", index_col="index")
st.dataframe(df)