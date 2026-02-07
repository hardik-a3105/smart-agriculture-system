from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# ---------- LOAD MODEL ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, 'models/model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'rb'))

# ---------- NAVIGATION ----------
@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

@app.route('/fertilizer.html')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/help.html')
def help_page():
    return render_template('help.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

@app.route('/yield.html')
def yield_page():
    return render_template('yield.html')

# ---------- CROP PREDICTION ----------
@app.route('/crop.html')
def crop_page():
    return render_template('crop.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        N = float(request.form['n'])
        P = float(request.form['p'])
        K = float(request.form['k'])
        temp = float(request.form['temperature'])
        hum = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rain = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temp, hum, ph, rain]])
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        return render_template('crop.html', prediction=prediction.capitalize())

    except Exception as e:
        return render_template('crop.html', prediction="Invalid input! Please check values.")

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
