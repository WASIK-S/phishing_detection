import pandas as pd
import joblib
import os

print("Creating training_features_list.pkl ...")

# Load processed dataset
df = pd.read_csv("processed_features.csv")

# Remove target column if exists
possible_targets = ["Result", "label", "Label", "class"]

for col in possible_targets:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Removed target column: {col}")

# Get feature list
feature_list = list(df.columns)

print("Feature count:", len(feature_list))
print("Features:", feature_list)

# Save into models folder
os.makedirs("models", exist_ok=True)

joblib.dump(feature_list, "models/training_features_list.pkl")

print("✅ training_features_list.pkl created successfully!")
