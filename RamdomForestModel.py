import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
df = pd.read_csv(r"C:\Users\Gaurav Bile\Videos\1Study\SKY internship\IT Employee Performance Prediction Dashboard\HR_Employee_Preprocessed.csv", encoding='utf-8')
print("File loaded successfully!")

# Display columns to verify
print("Dataset Columns:", df.columns.tolist())

# Ensure the correct target column is used
if 'PerformanceRating' in df.columns:
    target_column = 'PerformanceRating'
elif 'Target_Attriction' in df.columns:
    target_column = 'Target_Attriction'
else:
    print("Error: No suitable target column found in the dataset!")
    exit()

# Convert categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df)

# Split data into features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest model with optimizations
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

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize a single decision tree
plt.figure(figsize=(20, 10))
plt.title("Employee Performance Decision Tree", fontsize=16, color='blue')

tree.plot_tree(
    rf_model.estimators_[0], 
    feature_names=X.columns,
    class_names=[str(c) for c in y.unique()], 
    filled=True,
    fontsize=8
)

plt.show()
