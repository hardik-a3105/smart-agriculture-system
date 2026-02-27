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
        state           = request.form["State"]
        crop            = request.form["Crop"]
        crop_year       = int(request.form["Crop_Year"])
        season          = request.form["Season"]
        area            = float(request.form["Area"])
        annual_rainfall = float(request.form["Annual_Rainfall"])
        fertilizer      = float(request.form["Fertilizer"])
        pesticide       = float(request.form["Pesticide"])

        # Validate ranges
        validate_range(area,            0, 100000, "Area")
        validate_range(annual_rainfall, 0,   5000, "Annual Rainfall")
        validate_range(fertilizer,      0,  10000, "Fertilizer")
        validate_range(pesticide,       0,   1000, "Pesticide")
        validate_range(crop_year,    1900,   2100, "Crop Year")

        # --- Use trained yield model if available ---
        if (yield_model is not None and le_crop_yield is not None
                and le_state_yield is not None and le_season_yield is not None):
            try:
                crop_enc   = le_crop_yield.transform([crop])[0]
                season_enc = le_season_yield.transform([season])[0]
                state_enc  = le_state_yield.transform([state])[0]

                # The model was trained with a Production column. Since users
                # cannot know production in advance, we estimate it using the
                # area and a reasonable median yield (≈ 1.0 ton/hectare).
                estimated_production = area * 1.0

                # Feature order must match training:
                # Crop, Crop_Year, Season, State, Area, Production,
                # Annual_Rainfall, Fertilizer, Pesticide
                input_arr = np.array([[
                    crop_enc, crop_year, season_enc, state_enc,
                    area, estimated_production,
                    annual_rainfall, fertilizer, pesticide
                ]])
                prediction = round(float(yield_model.predict(input_arr)[0]), 2)
                logger.info(
                    f"Yield predicted (model): {prediction}"
                    f" | State={state}, Crop={crop}, Season={season}"
                )
            except Exception as enc_err:
                logger.warning(
                    f"Yield model encoding failed, falling back to heuristic: {enc_err}"
                )
                prediction = _heuristic_yield(area, annual_rainfall, fertilizer)
        else:
            logger.info("Yield model not loaded, using heuristic estimation.")
            prediction = _heuristic_yield(area, annual_rainfall, fertilizer)

        return render_template("yield.html", yield_prediction=prediction)

    except ValueError as ve:
        logger.warning(f"Yield validation error: {ve}")
        return render_template("yield.html", yield_prediction=f"Input Error: {ve}", error=True)
    except Exception as e:
        logger.error(f"Yield prediction failed: {e}")
        return render_template(
            "yield.html",
            yield_prediction="Prediction failed. Please check your inputs.",
            error=True,
        )

def _heuristic_yield(area, annual_rainfall, fertilizer):
    """Simple heuristic fallback when the ML model is unavailable."""
    # Estimate yield (tons per hectare) from available data
    base_yield = 1.0
    # Rainfall factor: more rainfall generally helps, up to a point
    rainfall_factor = min(annual_rainfall / 1200.0, 2.0)
    # Fertilizer factor: more fertilizer generally boosts production
    fert_factor = 1.0 + min(fertilizer / 5000.0, 1.0)
    estimated_yield = base_yield * rainfall_factor * fert_factor
    # Return total production estimate (yield * area)
    return round(estimated_yield * area, 2)

# ================== RUN APP ==================
if __name__ == "__main__":
    app.run(debug=True)
