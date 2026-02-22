import logging
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import os

# ================== LOGGING SETUP ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='frontend')

# ================== LOAD MODELS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    """Load a model safely, log errors if the file is missing."""
    full_path = os.path.join(BASE_DIR, path)
    try:
        model = joblib.load(full_path)
        logger.info(f"Model loaded: {path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file NOT found: {full_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model {path}: {e}")
        return None

crop_model           = load_model("models/crop_recommendation_model.pkl")
fertilizer_model     = load_model("models/fertilizer_recommendation_model.pkl")
le_crop_fert         = load_model("models/le_crop_fertilizer.pkl")
le_soil_fert         = load_model("models/le_soil_fertilizer.pkl")
yield_model          = load_model("models/yield_prediction_model.pkl")
le_crop_yield        = load_model("models/le_crop_yield.pkl")
le_season_yield      = load_model("models/le_season_yield.pkl")
le_state_yield       = load_model("models/le_state_yield.pkl")

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

# ================== INPUT VALIDATION HELPERS ==================
def validate_range(value, min_val, max_val, name):
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}. Got: {value}")

# ================== CROP PREDICTION ==================
@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    if crop_model is None:
        logger.error("Crop model not loaded.")
        return render_template("crop.html", prediction="Model unavailable. Please contact admin.", error=True)

    try:
        N    = float(request.form["n"])
        P    = float(request.form["p"])
        K    = float(request.form["k"])
        temp = float(request.form["temperature"])
        hum  = float(request.form["humidity"])
        ph   = float(request.form["ph"])
        rain = float(request.form["rainfall"])

        # Validate plausible ranges
        validate_range(N,    0,   140, "Nitrogen (N)")
        validate_range(P,    5,   145, "Phosphorus (P)")
        validate_range(K,    5,   205, "Potassium (K)")
        validate_range(temp, 0,    50, "Temperature")
        validate_range(hum,  0,   100, "Humidity")
        validate_range(ph,   0,    14, "pH")
        validate_range(rain, 0,  5000, "Rainfall")

        input_data  = np.array([[N, P, K, temp, hum, ph, rain]])
        prediction  = crop_model.predict(input_data)[0]

        logger.info(f"Crop predicted: {prediction} | Inputs: N={N}, P={P}, K={K}, T={temp}, H={hum}, pH={ph}, R={rain}")
        return render_template("crop.html", prediction=prediction.capitalize())

    except ValueError as ve:
        logger.warning(f"Crop validation error: {ve}")
        return render_template("crop.html", prediction=f"Input Error: {ve}", error=True)
    except Exception as e:
        logger.error(f"Crop prediction failed: {e}")
        return render_template("crop.html", prediction="Prediction failed. Please check your inputs.", error=True)

# ================== FERTILIZER PREDICTION ==================
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    if fertilizer_model is None:
        logger.error("Fertilizer model not loaded.")
        return render_template("fertilizer.html", fertilizer_res="Model unavailable. Please contact admin.", error=True)

    try:
        temp     = float(request.form["temperature"])
        hum      = float(request.form["humidity"])
        moisture = float(request.form["moisture"])
        soil     = request.form["soil"]
        crop     = request.form["crop"]
        N        = float(request.form["n"])
        P        = float(request.form["p"])
        K        = float(request.form["k"])

        # Validate ranges
        validate_range(temp,     0,  60,  "Temperature")
        validate_range(hum,      0,  100, "Humidity")
        validate_range(moisture, 0,  100, "Moisture")
        validate_range(N,        0,  200, "Nitrogen (N)")
        validate_range(P,        0,  200, "Phosphorus (P)")
        validate_range(K,        0,  200, "Potassium (K)")

        # Encode categorical features if label encoders are loaded
        try:
            soil_enc = le_soil_fert.transform([soil])[0] if le_soil_fert else soil
            crop_enc = le_crop_fert.transform([crop])[0] if le_crop_fert else crop
        except ValueError as enc_err:
            logger.warning(f"Encoding error for fertilizer: {enc_err}")
            return render_template("fertilizer.html",
                                   fertilizer_res=f"Unknown crop or soil type: {enc_err}", error=True)

        input_arr = np.array([[temp, hum, moisture, soil_enc, crop_enc, N, K, P]])
        raw_pred  = fertilizer_model.predict(input_arr)[0]

        # Map numeric prediction to fertilizer name if needed
        FERTILIZER_NAMES = {
            0:  "10-26-26",
            1:  "14-35-14",
            2:  "17-17-17",
            3:  "20-20",
            4:  "28-28",
            5:  "DAP",
            6:  "Urea"
        }
        if isinstance(raw_pred, (int, np.integer)):
            prediction_label = FERTILIZER_NAMES.get(int(raw_pred), f"Fertilizer #{raw_pred}")
        else:
            prediction_label = str(raw_pred)

        logger.info(f"Fertilizer predicted: {prediction_label}")
        return render_template("fertilizer.html", fertilizer_res=prediction_label)

    except ValueError as ve:
        logger.warning(f"Fertilizer validation error: {ve}")
        return render_template("fertilizer.html", fertilizer_res=f"Input Error: {ve}", error=True)
    except Exception as e:
        logger.error(f"Fertilizer prediction failed: {e}")
        return render_template("fertilizer.html",
                               fertilizer_res="Prediction failed. Please check your inputs.", error=True)

# ================== YIELD PREDICTION ==================
@app.route("/predict_yield", methods=["POST"])
def predict_yield():
    try:
        region       = request.form["Region"]
        crop         = request.form["Crop"]
        soil_type    = request.form["Soil_Type"]
        rainfall     = float(request.form["Rainfall_mm"])
        temperature  = float(request.form["Temperature_Celsius"])
        weather      = request.form["Weather_Condition"]
        fert_used    = request.form["Fertilizer_Used"]
        irrig_used   = request.form["Irrigation_Used"]
        days_harvest = int(request.form["Days_to_Harvest"])

        # Validate ranges
        validate_range(rainfall,     0, 5000, "Rainfall")
        validate_range(temperature,  0,   60, "Temperature")
        validate_range(days_harvest, 1,  365, "Days to Harvest")

        # --- Use trained yield model if available ---
        if yield_model is not None and le_crop_yield is not None and le_state_yield is not None:
            try:
                crop_enc  = le_crop_yield.transform([crop])[0]
                state_enc = le_state_yield.transform([region])[0]
                fert_bin  = 1 if fert_used == "Yes" else 0
                irrig_bin = 1 if irrig_used == "Yes" else 0

                input_arr = np.array([[crop_enc, state_enc, rainfall, temperature, fert_bin, irrig_bin, days_harvest]])
                prediction = round(float(yield_model.predict(input_arr)[0]), 2)
                logger.info(f"Yield predicted (model): {prediction} | Region={region}, Crop={crop}")
            except Exception as enc_err:
                logger.warning(f"Yield model encoding failed, falling back to heuristic: {enc_err}")
                prediction = _heuristic_yield(rainfall, fert_used, irrig_used)
        else:
            logger.info("Yield model not loaded, using heuristic estimation.")
            prediction = _heuristic_yield(rainfall, fert_used, irrig_used)

        return render_template("yield.html", yield_prediction=prediction)

    except ValueError as ve:
        logger.warning(f"Yield validation error: {ve}")
        return render_template("yield.html", yield_prediction=f"Input Error: {ve}")
    except Exception as e:
        logger.error(f"Yield prediction failed: {e}")
        return render_template("yield.html", yield_prediction="Prediction failed. Please check your inputs.")

def _heuristic_yield(rainfall, fert_used, irrig_used):
    """Simple heuristic fallback when the ML model is unavailable."""
    base = 3.5
    if fert_used == "Yes":
        base += 0.8
    if irrig_used == "Yes":
        base += 0.5
    base += (rainfall / 1000) * 0.3
    return round(base, 2)

# ================== RUN APP ==================
if __name__ == "__main__":
    app.run(debug=True)
