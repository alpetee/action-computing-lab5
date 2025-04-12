"""
Binary classifier for the gener of names
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv('../classification-demo/breast-cancer.csv')

X = df[['radius_mean','texture_mean','compactness_mean','concavity_mean']]
y = df['diagnosis']

# Splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using logistic regression for binary classification
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Training the model and prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculating accuracy and ROC AUC
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
