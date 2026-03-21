<div align="center">

# 🌱 Smart Harvest — AI-Powered Agriculture System

**Smart, data-driven farming decisions for Indian farmers using Machine Learning**
</div>

---

## 📖 Overview

**Smart Harvest** is a full-stack, AI-driven agricultural advisory platform built to empower farmers to make informed, data-backed farming decisions using Machine Learning models trained on real agricultural datasets. 

The system offers three core AI advisors — **Crop Recommendation**, **Fertilizer Recommendation**, and **Yield Estimation** — all accessible through a modern, mobile-friendly web interface with user authentication and a dynamic history-tracking dashboard.

---

## 🏛️ Architecture

Smart Harvest utilizes a **Dual-Backend Microservice Architecture** for optimal performance and pure separation of concerns:

1. **Node.js Auth & Frontend Server**: Handles secure user registration, login, profile management, and serves the static frontend files.
2. **Python Flask Machine Learning API**: A pure REST API serving predictions via `joblib` serialized scikit-learn models. Cross-Origin Resource Sharing (CORS) is explicitly configured to allow the frontend to securely interact with the ML API.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌾 **Crop Recommendation** | Suggests optimal crops based on soil NPK levels, temperature, humidity, pH, and rainfall. |
| 🧪 **Fertilizer Recommendation** | Recommends fertilizer type based on soil health, moisture, and crop type. |
| 📈 **Yield Prediction** | Forecasts expected production (Tons/Hectare) using weather, irrigation, and management inputs. |
| ⚡ **Asynchronous AI** | Real-time JS `fetch` integration with loading spinners for seamless user experience without page reloads. |
| 🔐 **User Authentication** | Secure farmer registration & login with bcrypt password hashing + MongoDB storage. |
| 📊 **Dynamic Dashboard** | Personalised dashboard showing farm profile, dynamic estimated revenue, and trackable prediction history. |
| 📝 **History Management** | "View All" and "Clear History" capabilities to manage past AI predictions. |
| 📱 **Responsive UI** | Beautiful Glassmorphism UI using Bootstrap 5 + Poppins typography — works on mobile, tablet, and desktop. |

---

## 🧠 Machine Learning Models

| Task | Algorithm | Output |
|---|---|---|
| Crop Recommendation | Random Forest **Classifier** | Crop name (e.g. Rice, Wheat, Cotton) |
| Fertilizer Recommendation | Random Forest **Classifier** | Fertilizer name (e.g. Urea, DAP, 17-17-17) |
| Yield Prediction | Random Forest **Regressor** | Expected Yield Production in Tons |

All models are trained in Jupyter notebooks, serialized with `joblib`, and served via the Flask REST API.

---

## 🗂️ Project Structure

```
Smart Agri-system/
│
├── app.py                        # Python Model API — ML prediction routes (Port 5000)
│
├── backend/
│   └── server.js                 # Node.js Auth/Frontend Server (Port 3000)
│
├── frontend/                     # Static HTML, CSS, JS Frontend Assets
│   ├── index.html                # Landing page
│   ├── dashboard.html            # Farmer Dashboard with History Module
│   ├── crop.html                 # Crop Recommendation UI
│   ├── fertilizer.html           # Fertilizer Recommendation UI
│   ├── yield.html                # Yield Prediction UI
│   ├── ...                       # Other pages (Login, Register, Profile, etc.)
│
├── models/                       # Trained ML model files (.pkl)
│
├── data/                         # Source datasets (CSV) for Jupyter training
│
├── notebooks/                    # Jupyter notebooks for model training pipelines
│
├── render.yaml                   # Infrastructure as Code (IaC) deployment script for Render.com
├── .env                          # Environment variables (MongoDB UI, Ports)
├── package.json                  # Node.js dependencies
└── requirements.txt              # Python dependencies (Flask, scikit-learn, Flask-CORS)
```

---

## ⚙️ Tech Stack

**Machine Learning API (Python)**
- `Flask` & `Flask-CORS` — REST API
- `scikit-learn` — Random Forest models
- `NumPy` & `Pandas` — Data processing

**Auth & Frontend Server (Node.js)**
- `Express.js` — Auth & static asset server
- `Mongoose` — MongoDB ODM
- `bcryptjs` — Password hashing

**Frontend**
- `Vanilla JS`, `HTML5`, `CSS3`
- `Bootstrap 5.3` & `Bootstrap Icons`
- `Animate.css`

**Database & Deployment**
- `MongoDB Atlas` — Cloud-hosted user data & history
- `Render.com` — Cloud Application Hosting (via `render.yaml`)

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

### 2. Python Environment Setup (ML API)

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

---

### 3. Node.js Setup (Auth Server / Frontend Loader)

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
*(Update the MongoDB URI with your valid Atlas connection string)*

---

### 5. Run the Application

The system requires both backend servers running simultaneously.

**Terminal 1 — Flask (ML API):**
```bash
python app.py
# Runs on Port 5000 by default
```

**Terminal 2 — Node.js (Auth/Frontend API):**
```bash
node backend/server.js
# Runs on Port 3000 by default
```

Open your browser to: **[http://localhost:3000](http://localhost:3000)**

---

## ☁️ Deployment (Render)

This repository includes a `render.yaml` Blueprint file mapped to automatically deploy two separate services to Render.com:
1. **Node.js Web Service**: Hosts the frontend assets and authorization API.
2. **Python Web Service**: Hosts the Machine Learning API.

To deploy, simply link your GitHub repository to your Render Dashboard and apply the `render.yaml` Blueprint block.

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

<div align="center">

**Built with ❤️ for Indian Farmers**

</div>