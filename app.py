import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# 1. Load Model + Columns
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model", "columns.pkl"))

# Extract locations
locations = [col for col in columns if col not in ['total_sqft', 'bath', 'bhk']]

# -----------------------------
# 2. UI
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 Bengaluru House Price Predictor")
st.write("Enter property details to estimate price")

# Inputs
total_sqft = st.number_input("Total Sqft", 300, 10000, 1200)
bath = st.slider("Bathrooms", 1, 10, 2)
bhk = st.slider("BHK", 1, 10, 2)

location = st.selectbox("Location", sorted(locations))

# -----------------------------
# 3. Prediction Function
# -----------------------------
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))

    # numeric features
    x[columns.get_loc('total_sqft')] = sqft
    x[columns.get_loc('bath')] = bath
    x[columns.get_loc('bhk')] = bhk

    # location (one-hot)
    if location in columns:
        x[columns.get_loc(location)] = 1

    return model.predict([x])[0]

# -----------------------------
# 4. Predict Button
# -----------------------------
if st.button("Predict Price"):
    try:
        price = predict_price(location, total_sqft, bath, bhk)
        st.success(f"💰 Estimated Price: ₹ {price:.2f} Lakhs")
    except Exception as e:
        st.error(f"Error: {e}")