import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

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
def pltfeaturesDistribution(df):
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
def pltboxDistribution(df):
    fig, axes = plt.subplots(5,5, figsize=(20,10))
    axes = axes.flatten()

    for i, feature in enumerate(df.columns):
        pltBox(axes[i], df, feature)

    plt.tight_layout()
    st.pyplot(fig)