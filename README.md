# 🌱 Smart Harvest – Smart Agriculture System using Machine Learning

Smart Harvest is a Machine Learning–based smart agriculture system designed to assist farmers in making **data-driven decisions**.  
The system provides **crop recommendation**, **fertilizer recommendation**, and **crop yield prediction** based on soil parameters, weather conditions, and historical agricultural data.
This project focuses on applying ML algorithms to real-world agricultural problems, making farming smarter, more efficient, and sustainable.

---

# 🚧 Project Status

This project is actively under development.

---

## 📌 Features

- 🌾 **Crop Recommendation**
  - Suggests the most suitable crop based on soil nutrients and environmental conditions.

- 🧪 **Fertilizer Recommendation**
  - Recommends appropriate fertilizer depending on soil health and crop type.

- 📊 **Crop Yield Prediction**
  - Predicts expected crop yield (tons/hectare) using weather, soil, and farming inputs.

- 🌐 **Web Interface (UI)**
  - User-friendly frontend for farmers and users to input data and view predictions.

---

## 🧠 Machine Learning Models Used

- **Random Forest Classifier**
  - Used for crop recommendation
  - Used for fertilizer recommendation

- **Random Forest Regressor**
  - Used for crop yield prediction

These models were chosen for their robustness, ability to handle non-linear data, and good performance on tabular datasets.

---

## 🗂️ Project Structure
```
Smart-Harvest-ML/
│
├── data/
│ ├── crop_data.csv
│ ├── fertilizer_data.csv
│ └── yield_data.csv
│
├── notebooks/
│ ├── crop_recommendation.ipynb
│ ├── fertilizer_recommendation.ipynb
│ └── yield_prediction.ipynb
│
├── models/
│ ├── crop_model.pkl
│ ├── fertilizer_model.pkl
│ ├── yield_model.pkl
│ └── scaler.pkl
│
├── app/
│ ├── main.py
│ └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## 📥 Dataset Description

The datasets used in this project include features such as:

- Soil nutrients (N, P, K)
- Temperature
- Humidity
- Rainfall
- Soil type
- Crop type
- Fertilizer usage
- Irrigation status
- Weather conditions
- Days to harvest

All datasets are preprocessed before training, including:
- Handling missing values
- Encoding categorical features
- Feature scaling

---

## ⚙️ Technologies Used

- **Programming Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - Joblib / Pickle  
- **Backend:** Flask / FastAPI  
- **Frontend:** HTML, CSS, Bootstrap  
- **Version Control:** Git & GitHub

---

## ⭐ Support

If you find this project useful, please consider giving it a ⭐ on GitHub or contributing to its development.
