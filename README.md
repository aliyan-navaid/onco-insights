# Onco-Insights

End-to-end cancer severity prediction using KNN, built with a Streamlit dashboard. Features a complete Data Science workflow: data cleaning, EDA, feature selection, model training, and evaluation.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-3b6da0?style=flat&logo=seaborn&logoColor=white)](https://seaborn.pydata.org)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **Data Pipeline**: Load, inspect, clean, and preprocess patient data interactively
- **EDA & Feature Selection**: Visualize distributions and select features via correlation thresholds
- **KNN Model**: Train, evaluate, and serialize models with adjustable hyperparameters
- **Streamlit Dashboard**: Optimized with `@st.cache_data` and `st.session_state` for responsive UX
- **Model Persistence**: Save/load trained models via Joblib for reuse

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12+ |
| Dashboard | Streamlit |
| ML | Scikit-Learn (KNN, StandardScaler) |
| Data | Pandas, NumPy |
| Viz | Matplotlib, Seaborn |
| Serialization | Joblib |

---

## Quick Start

```bash
# Clone repo
git clone https://github.com/aliyan-navaid/onco-insights.git
cd onco-insights

# Setup environment
python -m venv venv && source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install & run
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

Source: [Cancer Patients Data (Kaggle)](https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data)

| Feature Type | Examples |
|--------------|----------|
| Demographics | Age, Gender |
| Risk Factors | Air Pollution, Smoking, Genetic Risk, Occupational Hazards |
| Health Metrics | Obesity, Chronic Lung Disease, Balanced Diet |
| Target | `Level` (Low / Medium / High severity) |

---

## Highlights

- **Caching**: `@st.cache_data` decorators on data loading & plotting functions to minimize recomputation
- **Session State**: Manages training flow (`firstTrain`, `reTrain` flags) across Streamlit reruns
- **Modular Design**: Separated logic in `utils/` module for maintainability and testing
- **Joblib Serialization**: Persist trained KNN models (`model.joblib`) for instant reload without retraining