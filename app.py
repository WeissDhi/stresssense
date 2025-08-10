from flask import Flask, render_template, request
import pandas as pd
from catboost import CatBoostClassifier
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model and columns
model = CatBoostClassifier()
model.load_model("catboost_depression_model.cbm")
model_columns = joblib.load("model_columns.pkl")  # pastikan file ini ada

def preprocess_input(form):
    # Log form data
    logger.debug("Form data received: %s", dict(form))
    
    data = {
        "id": 0,
        "Gender": form["gender"],
        "Age": int(form["age"]),
        "Profession": "Student",
        "Academic Pressure": int(form["academic_pressure"]),
        "Work Pressure": int(form["work_pressure"]),
        "CGPA": (float(form["cgpa"]) / 4) * 10,
        "Study Satisfaction": 3,
        "Job Satisfaction": 0,
        "Sleep Duration": float(form["sleep_duration"]),
        "Dietary Habits": form["diet"],  # Ganti dari Diet
        "Degree": "BSc",
        "Have you ever had suicidal thoughts ?": form["suicidal_thoughts"],
        "Work/Study Hours": 5,
        "Financial Stress": int(form["financial_stress"]),
        "Family History of Mental Illness": form["family_history"],
        "Total_Pressure": int(form["academic_pressure"]) + int(form["work_pressure"]),
        # Sleep_Quality logic
    }

    # Sleep Quality calculation
    if data["Sleep Duration"] < 5:
        data["Sleep_Quality"] = "Poor"
    elif 5 <= data["Sleep Duration"] <= 8:
        data["Sleep_Quality"] = "Normal"
    else:
        data["Sleep_Quality"] = "Over"

    # Tambahkan kolom yang mungkin tidak ada
    for col in model_columns:
        if col not in data:
            data[col] = 0
    
    df = pd.DataFrame([data])
    df = df[model_columns]
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}
    if request.method == "POST":
        logger.debug("Request method: POST")
        logger.debug("Request form data: %s", dict(request.form))
        logger.debug("Request files: %s", dict(request.files))
        form_data = request.form.to_dict()
        try:
            input_df = preprocess_input(request.form)
            pred_prob = model.predict_proba(input_df)[0][1]
            prediction = round(pred_prob * 100, 2)
        except Exception as e:
            logger.error("Error processing form: %s", str(e), exc_info=True)
            raise
    return render_template("index.html", prediction=prediction, form_data=form_data)

@app.route("/edukasi")
def edukasi():
    return render_template("edukasi.html")

@app.route("/direktori")
def direktori():
    return render_template("direktori.html")

@app.route("/tentang")
def tentang():
    return render_template("tentang.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
