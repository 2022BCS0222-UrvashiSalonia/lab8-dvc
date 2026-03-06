import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import os

# Load data
df = pd.read_csv("data/housing.csv")
print(f"Dataset size: {len(df)} rows")

# Features and target
X = df.drop("median_house_value", axis=1)
X = pd.get_dummies(X)  # handle categorical column 'ocean_proximity'
y = df["median_house_value"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")
print(f"Training samples: {len(X_train)}")

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics = {
    "rmse": round(rmse, 2),
    "r2": round(r2, 4),
    "dataset_size": len(df),
    "training_samples": len(X_train)
}
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to metrics/metrics.json")