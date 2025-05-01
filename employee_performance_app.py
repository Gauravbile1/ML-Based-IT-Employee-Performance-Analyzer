import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import streamlit as st
import seaborn as sns

# Load dataset
df = pd.read_csv("HR_Employee_Preprocessed.csv", encoding='utf-8')

# Ensure the correct target column is used
if 'PerformanceRating' in df.columns:
    target_column = 'PerformanceRating'
elif 'Target_Attriction' in df.columns:
    target_column = 'Target_Attriction'
else:
    st.error("No suitable target column found in the dataset!")
    exit()

# Convert categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df)

# Handle duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Add two important columns for project goals
df['WorkHoursPerWeek'] = np.random.randint(30, 60, size=len(df))
df['TrainingScore'] = np.random.randint(50, 100, size=len(df))

# Split data into features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=25,
    max_depth=5,
    min_samples_split=10,
    bootstrap=True,
    max_samples=0.8,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Streamlit Web App
st.set_page_config(page_title="IT Employee Performance Prediction", layout="wide")
st.title("💼 IT Employee Performance Prediction Dashboard")

# Sidebar
st.sidebar.header("Project Overview")
st.sidebar.write("**Goal:** Predict employee performance based on multiple factors like involvement, skills, and feedback.")
st.sidebar.write("**Tech Stack:** Python (Pandas, Scikit-learn), Streamlit for visualization")
st.sidebar.write("**Dataset:** HR Analytics Dataset")

# Performance Metrics
st.header("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
col2.text("Classification Report")
col2.text(classification_report(y_test, y_pred))

# Decision Tree Visualization
st.header("🌳 Decision Tree Visualization")
plt.figure(figsize=(15, 8))
plt.title("Employee Performance Decision Tree", fontsize=18, color='#4CAF50')
tree.plot_tree(
    rf_model.estimators_[0], 
    feature_names=X.columns,
    class_names=[str(c) for c in y.unique()], 
    filled=True,
    fontsize=10
)
st.pyplot(plt)

# Alternative Data Visualization (Bar Plot)
st.header("📌 Employee Performance Analysis")
plt.figure(figsize=(12, 8))
performance_counts = df[target_column].value_counts().sort_index()
sns.barplot(x=performance_counts.index, y=performance_counts.values, palette="viridis")
plt.title("Employee Performance Distribution", fontsize=16, color='#4CAF50')
plt.xlabel("Performance Rating")
plt.ylabel("Count")
st.pyplot(plt)

# Add slicers
st.sidebar.header("🔍 Data Slicer")

# Three specific slicers for key features
slicer_columns = ['Department_Sales', 'Department_R&D', 'Department_HR']
for idx, col in enumerate(slicer_columns):
    if col in df.columns:
        unique_values = [col.replace('Department_', '') for col in slicer_columns]
        selected_values = st.sidebar.multiselect(
            "Select Department", options=unique_values, default=unique_values, key=f"slicer_{idx}"
        )
        df = df[df[slicer_columns].sum(axis=1).astype(bool)]

st.write("Filtered Data Sample:", df.head())

# Add a new visualization - Boxplot
st.header("📌 Salary vs Performance")
if 'MonthlyIncome' in df.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x=target_column, y='MonthlyIncome', palette="coolwarm")
    plt.title("Salary Distribution by Performance", fontsize=16, color='#4CAF50')
    plt.xlabel("Performance Rating")
    plt.ylabel("Monthly Income")
    st.pyplot(plt)

st.success("✨ Dashboard loaded successfully!")
