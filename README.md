<div align="center">

# 🌱 Smart Harvest — AI-Powered Agriculture System

**Smart, data-driven farming decisions for Indian farmers using Machine Learning**
</div>

---

## 📖 Overview

**Smart Harvest** is a full-stack, AI-driven agricultural advisory platform built for the **Smart India Hackathon (SIH)**. It empowers farmers in **Gujarat, Jharkhand, and Maharashtra** to make informed, data-backed farming decisions using Machine Learning models trained on real agricultural datasets.

The system offers three core AI advisors — crop recommendation, fertilizer recommendation, and yield estimation — all accessible through a modern, mobile-friendly web interface with user authentication.

---

## ✨ Features

| Feature | Description |
|---|---|
| � **Crop Recommendation** | Suggests the optimal crop based on soil NPK levels, temperature, humidity, pH, and rainfall |
| 🧪 **Fertilizer Recommendation** | Recommends fertilizer type based on soil health, moisture, and crop type |
| � **Yield Prediction** | Forecasts expected production (Tons/Hectare) using weather, irrigation, and management inputs |
| 🔐 **User Authentication** | Secure farmer registration & login with bcrypt password hashing + MongoDB storage |
| 📊 **Farmer Dashboard** | Personalised dashboard showing farm profile, quick stats, and recent activity |
| 👤 **Profile Management** | View and update personal and farm details |
| 📱 **Responsive UI** | Bootstrap 5 + Poppins typography — works on mobile, tablet, and desktop |

---

## 🧠 Machine Learning Models

| Task | Algorithm | Output |
|---|---|---|
| Crop Recommendation | Random Forest **Classifier** | Crop name (e.g. Rice, Wheat, Cotton) |
| Fertilizer Recommendation | Random Forest **Classifier** | Fertilizer name (e.g. Urea, DAP, 17-17-17) |
| Yield Prediction | Random Forest **Regressor** | Yield in Tons/Hectare |

All models are trained in Jupyter notebooks, serialized with `joblib`, and served via a Flask REST API.

---

## 🗂️ Project Structure

```
Smart Agri-system/
│
├── app.py                        # Flask application — ML prediction routes
│
├── backend/
│   └── server.js                 # Node.js + Express — auth API (login/register)
│
├── frontend/                     # All HTML pages
│   ├── index.html                # Landing page
│   ├── crop.html                 # Crop Recommendation UI
│   ├── fertilizer.html           # Fertilizer Recommendation UI
│   ├── yield.html                # Yield Prediction UI
│   ├── dashboard.html            # Farmer Dashboard
│   ├── profile.html              # Profile Management
│   ├── login.html                # Login Page
│   ├── register.html             # Registration Page
│   ├── about.html                # About the Project
│   ├── help.html                 # Help & FAQ
│   └── contact.html              # Contact Support
│
├── models/                       # Trained ML model files (.pkl)
│   ├── crop_recommendation_model.pkl
│   ├── fertilizer_recommendation_model.pkl
│   ├── le_crop_fertilizer.pkl
│   ├── le_soil_fertilizer.pkl
│   ├── yield_prediction_model.pkl
│   ├── le_crop_yield.pkl
│   ├── le_season_yield.pkl
│   └── le_state_yield.pkl
│
├── data/                         # Source datasets (CSV)
│   ├── Crop_recommendation.csv
│   ├── Fertilizer_Prediction.csv
│   └── yield_data.csv
│
├── notebooks/                    # Model training notebooks
│   ├── crop_recommendatiom.ipynb
│   ├── fertilizer_recommendation.ipynb
│   └── yield_prediction.ipynb
│
├── .env                          # Environment variables (not committed)
├── package.json                  # Node.js dependencies
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

**Machine Learning & Backend (Python)**
- `Flask` — REST API for ML predictions
- `scikit-learn` — Random Forest models
- `NumPy` & `Pandas` — Data processing
- `Joblib` — Model serialization

**Backend (Node.js)**
- `Express.js` — Auth API server
- `Mongoose` — MongoDB ODM
- `bcryptjs` — Password hashing
- `CORS`, `dotenv`

**Frontend**
- `HTML5`, `Vanilla CSS`, `JavaScript`
- `Bootstrap 5.3`
- `Bootstrap Icons`
- `Animate.css`
- `Google Fonts (Poppins)`

**Database**
- `MongoDB Atlas` — Cloud-hosted user data

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB Atlas account (or local MongoDB)

---

### 1. Clone the Repository

```bash
git clone https://github.com/hardik-a3105/smart-agriculture-system.git
cd smart-agriculture-system
```

---

### 2. Python Environment Setup (Flask + ML Models)

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

---

### 3. Node.js Setup (Auth Server)

```bash
npm install
```

---

### 4. Environment Variables

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/smartagri
NODE_ENV=development
PORT=3000
```

---

### 5. Train the Models (if .pkl files are missing)

Open and run each notebook in order:

```
notebooks/crop_recommendatiom.ipynb
notebooks/fertilizer_recommendation.ipynb
notebooks/yield_prediction.ipynb
```

This will generate all `.pkl` files inside the `models/` directory.

---

### 6. Run the Application

**Terminal 1 — Flask (ML API, port 5000):**
```bash
python app.py
```

**Terminal 2 — Node.js (Auth API, port 3000):**
```bash
node backend/server.js
```

Then open your browser at: **[http://localhost:5000](http://localhost:5000)**

---

## 📥 Input Features Guide

### 🌱 Crop Recommendation

| Parameter | Range | Unit |
|---|---|---|
| Nitrogen (N) | 0 – 140 | kg/ha |
| Phosphorus (P) | 5 – 145 | kg/ha |
| Potassium (K) | 5 – 205 | kg/ha |
| Temperature | 0 – 50 | °C |
| Humidity | 0 – 100 | % |
| Soil pH | 0 – 14 | — |
| Rainfall | 0 – 5000 | mm |

### 🧪 Fertilizer Recommendation

Soil Type, Crop Type, NPK levels, Temperature, Humidity, Soil Moisture

### 📈 Yield Prediction

Region, Crop, Soil Type, Rainfall, Temperature, Weather Condition, Fertilizer & Irrigation usage, Days to Harvest

---

## 📸 Screenshots

> _Coming soon — live deployment screenshots_

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If Smart Harvest helped you or you found it interesting, please consider giving it a **⭐ star** on GitHub — it helps others discover the project!

<div align="center">

**Built with ❤️ for Indian Farmers | Smart India Hackathon (SIH) Project**

</div>
