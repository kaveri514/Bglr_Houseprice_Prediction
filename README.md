# 🏠 Bengaluru House Price Prediction

A simple Machine Learning web app that predicts house prices in Bengaluru based on property details.

---

## 🚀 Live App

Add your deployed link here:
👉 [https://bglrhousepriceprediction-lifs6xi93ca5agyjdcqzdc.streamlit.app/]

---

## 📌 Project Overview

This project predicts house prices using features like:

* Total Sqft
* BHK
* Bathrooms
* Location

The model is trained on a real Bengaluru housing dataset and deployed using Streamlit.

---

## 🧠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

---

## 🧠 Machine Learning Details

* **Algorithm Used:** XGBoost Regressor
* **Problem Type:** Regression

### 📊 Model Performance

* **R² Score:** 0.51
* **MAE:** Moderate prediction error

👉 The current model serves as a **baseline implementation**.
Performance can be improved further using advanced feature engineering and better data preprocessing.


---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python train_model.py

streamlit run app.py
```

---

## 📁 Project Structure

```
Bglr_HousePrice_prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── data/
│   └── Bengaluru_House_Data.csv
│
├── model/
│   ├── model.pkl
│   └── columns.pkl
```

---

## 🙌 Acknowledgements

Dataset taken from publicly available Bengaluru housing data (Kaggle).

---

## 📬 Contact

Your Name
Your Email
GitHub: https://github.com/your-username
