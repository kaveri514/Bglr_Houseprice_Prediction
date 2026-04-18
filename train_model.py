import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# -----------------------------
# 1. Load Dataset
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "data", "Bengaluru_House_Data.csv")

df = pd.read_csv(data_path)

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df = df.dropna()

# Convert BHK
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert total_sqft
def convert_sqft(x):
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna()

# Price per sqft
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# -----------------------------
# 3. Feature Selection
# -----------------------------
df = df[['location', 'total_sqft', 'bath', 'bhk', 'price']]

# Clean location (reduce noise)
df['location'] = df['location'].apply(lambda x: x.strip())
location_stats = df['location'].value_counts()

# Keep only popular locations
locations_less_than_10 = location_stats[location_stats <= 10]
df['location'] = df['location'].apply(
    lambda x: 'other' if x in locations_less_than_10 else x
)

# -----------------------------
# 4. One-Hot Encoding
# -----------------------------
dummies = pd.get_dummies(df['location'])
df = pd.concat([df, dummies], axis=1)
df.drop('location', axis=1, inplace=True)

# -----------------------------
# 5. Split Data
# -----------------------------
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Train Model
# -----------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)

model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate
# -----------------------------
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)

print(f"✅ Model trained successfully")
print(f"📊 R2 Score: {r2:.3f}")

# -----------------------------
# 8. Save Model + Columns
# -----------------------------
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")

joblib.dump(model, model_path)
joblib.dump(X.columns, columns_path)

print("✅ Model saved!")