import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Feature Selection
X = df[['Age', 'Sleep Duration']]
y = df['Sleep Recommendation']  # This column should contain labels like "Increase sleep", "Maintain schedule", etc.

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save models
joblib.dump(rf_model, "sleep_recommendation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Recommendation Model saved successfully!")
