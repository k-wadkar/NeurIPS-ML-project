import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
data = pd.read_csv("wdbc.csv") 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
# Handle missing values (if any)
def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity
# Separate features and target variable
X = data.drop(columns=['Diagnosis'])  # Assuming 'diagnosis' is the target column
y = data['Diagnosis']

# Normalize/Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)


# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
sensitivity_rf, specificity_rf = calculate_metrics(y_test, y_pred)
print("Random Forest: ")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(f"Sensitivity (True Positive Rate): {sensitivity_rf:.4f}")
print(f"Specificity (True Negative Rate): {specificity_rf:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
sensitivity_nb, specificity_nb = calculate_metrics(y_test, y_pred_nb)
print("Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(f"Sensitivity: {sensitivity_nb:.4f}")
print(f"Specificity: {specificity_nb:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
kernel_svm_model = SVC(kernel='rbf', random_state=42)

# Train the model
kernel_svm_model.fit(X_train, y_train)

# Make predictions
y_pred_kernel_svm = kernel_svm_model.predict(X_test)

# Evaluate the model
sensitivity_kernel_svm, specificity_kernel_svm = calculate_metrics(y_test, y_pred_kernel_svm)
print("Kernel Support Vector Machine (SVM):")
print("Accuracy:", accuracy_score(y_test, y_pred_kernel_svm))
print(f"Sensitivity: {sensitivity_kernel_svm:.4f}")
print(f"Specificity: {specificity_kernel_svm:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_kernel_svm))
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
sensitivity_svm, specificity_svm = calculate_metrics(y_test, y_pred_svm)
print("Support Vector Machine (SVM):")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(f"Sensitivity: {sensitivity_svm:.4f}")
print(f"Specificity: {specificity_svm:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))


# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
sensitivity_lr, specificity_lr = calculate_metrics(y_test, y_pred_lr)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(f"Sensitivity: {sensitivity_lr:.4f}")
print(f"Specificity: {specificity_lr:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
