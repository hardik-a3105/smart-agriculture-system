from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder='frontend')

# ================== LOAD MODELS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

crop_model = joblib.load(os.path.join(BASE_DIR, "models/crop_recommendation_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
fertilizer_model = joblib.load(
    os.path.join(BASE_DIR, "models/fertilizer_recommendation_pipeline.pkl")
)
fertilizer_encoder = joblib.load(
    os.path.join(BASE_DIR, "models/fertilizer_target_encoder.pkl")
)

# ================== HOME & PAGES ==================
@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/contact.html")
def contact():
    return render_template("contact.html")

@app.route("/dashboard.html")
def dashboard():
    return render_template("dashboard.html")

@app.route("/fertilizer.html")
def fertilizer():
    return render_template("fertilizer.html")

@app.route("/help.html")
def help_page():
    return render_template("help.html")

@app.route("/login.html")
def login():
    return render_template("login.html")

@app.route("/profile.html")
def profile():
    return render_template("profile.html")

@app.route("/yield.html")
def yield_page():
    return render_template("yield.html")

@app.route("/register.html")
def register():
    return render_template("register.html")

@app.route("/crop.html")
def crop_page():
    return render_template("crop.html")

# ================== CROP PREDICTION ==================
@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        N = float(request.form["n"])
        P = float(request.form["p"])
        K = float(request.form["k"])
        temp = float(request.form["temperature"])
        hum = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rain = float(request.form["rainfall"])

        input_data = np.array([[N, P, K, temp, hum, ph, rain]])
        scaled_data = scaler.transform(input_data)

        prediction = crop_model.predict(scaled_data)[0]

        return render_template("crop.html", prediction=prediction.capitalize())

    except Exception as e:
        print("CROP ERROR:", e)
        return render_template("crop.html", prediction="Invalid input!")

# ================== FERTILIZER PREDICTION ==================
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        # ---- Read Inputs ----
        temp = float(request.form["temperature"])
        hum = float(request.form["humidity"])
        moisture = float(request.form["moisture"])
        soil = request.form["soil"]
        crop = request.form["crop"]
        N = float(request.form["n"])
        P = float(request.form["p"])
        K = float(request.form["k"])

        # ---- Create DataFrame for Pipeline ----
        input_df = pd.DataFrame([{
            'Temperature': temp,
            'Humidity': hum,
            'Moisture': moisture,
            'Soil_Type': soil,
            'Crop_Type': crop,
            'Nitrogen': N,
            'Potassium': K,
            'Phosphorous': P
        }])

        # ---- Predict & Decode ----
        prediction_idx = fertilizer_model.predict(input_df)[0]
        prediction_label = fertilizer_encoder.inverse_transform([prediction_idx])[0]

        return render_template(
            "fertilizer.html",
            fertilizer_res=prediction_label
        )

    except Exception as e:
        print("FERTILIZER ERROR:", e)
        return render_template(
            "fertilizer.html",
            fertilizer_res="Invalid input! Please check values."
        )

# ================== YIELD PREDICTION ==================
@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    try:
        region = request.form["Region"]
        crop = request.form["Crop"]
        soil_type = request.form["Soil_Type"]
        rainfall = float(request.form["Rainfall_mm"])
        temperature = float(request.form["Temperature_Celsius"])
        weather = request.form["Weather_Condition"]
        fertilizer_used = request.form["Fertilizer_Used"]
        irrigation_used = request.form["Irrigation_Used"]
        days_to_harvest = int(request.form["Days_to_Harvest"])

        # Placeholder: return a simple estimation until a yield model is trained
        base_yield = 3.5
        if fertilizer_used == "Yes":
            base_yield += 0.8
        if irrigation_used == "Yes":
            base_yield += 0.5
        base_yield += (rainfall / 1000) * 0.3
        prediction = round(base_yield, 2)

        return render_template("yield.html", yield_prediction=prediction)

    except Exception as e:
        print("YIELD ERROR:", e)
        return render_template("yield.html", yield_prediction="Invalid input!")

# ================== RUN APP ==================
if __name__ == "__main__":
    app.run(debug=True)
