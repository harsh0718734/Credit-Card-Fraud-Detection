import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
# Step 2: Load the dataset
data = pd.read_csv("creditcard.csv")

print("First 5 rows of data:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nSummary statistics:")
print(data.describe())

print("\n--- Class Distribution ---")
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlierFraction = len(fraud) / float(len(valid))

print("Outlier fraction (fraud/valid):", outlierFraction)
print("Fraud cases:", len(fraud))
print("Valid cases:", len(valid))

print("\nAmount stats for FRAUD transactions:")
print(fraud['Amount'].describe())

print("\nAmount stats for VALID transactions:")
print(valid['Amount'].describe())

print("\nGenerating correlation heatmap...")

corr = data.corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

X = data.drop(['Class'], axis=1)
y = data['Class']

xData = X.values
yData = y.values

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", xTrain.shape)
print("Test shape:", xTest.shape)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
rfc.fit(xTrain, yTrain)

yPred = rfc.predict(xTest)

accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("MCC:", mcc)

conf_matrix = confusion_matrix(yTest, yPred)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
