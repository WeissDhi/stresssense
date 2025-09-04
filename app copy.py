from flask import Flask, render_template, request
import pandas as pd
from catboost import CatBoostClassifier
import joblib
import logging
import os
from math import exp

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model and columns
model = CatBoostClassifier()
model.load_model("catboost_depression_model.cbm")
model_columns = joblib.load("model_columns.pkl")  # pastikan file ini ada

def _safe_clip(value: float, low: float = 1e-6, high: float = 1 - 1e-6) -> float:
    return max(low, min(high, value))


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
        # Jika fitur ini dipakai di model, nilai default yang lebih netral akan membantu variasi
        "Study Satisfaction": 3,
        "Job Satisfaction": 3,
        "Sleep Duration": float(form["sleep_duration"]),
        "Dietary Habits": form["diet"],  # Ganti dari Diet
        "Degree": "BSc",
        "Have you ever had suicidal thoughts ?": form["suicidal_thoughts"],
        "Work/Study Hours": 6,
        "Financial Stress": int(form["financial_stress"]),
        "Family History of Mental Illness": form["family_history"],
        "Total_Pressure": int(form["academic_pressure"]) + int(form["work_pressure"]),
        # Sleep_Quality logic
    }

    # Optional mapping untuk field biner tambahan dari form (jika memang ada di model)
    # Hanya tambahkan jika kolom tersebut memang ada di model_columns
    try:
        if "Financial Problem" in model_columns and "financial_problem" in form:
            data["Financial Problem"] = "Yes" if form["financial_problem"] == "Yes" else "No"
        if ("Health Issue" in model_columns or "Health Issues" in model_columns) and "health_issue" in form:
            col_name = "Health Issue" if "Health Issue" in model_columns else "Health Issues"
            data[col_name] = "Yes" if form["health_issue"] == "Yes" else "No"
    except Exception as e:
        logger.warning("Optional binary fields mapping failed: %s", str(e))

    # Sleep Quality calculation
    if data["Sleep Duration"] < 5:
        data["Sleep_Quality"] = "Poor"
    elif 5 <= data["Sleep Duration"] <= 8:
        data["Sleep_Quality"] = "Normal"
    else:
        data["Sleep_Quality"] = "Over"

    # Tambahkan kolom yang mungkin tidak ada dengan default yang lebih tepat secara tipe
    default_values = {
        # Numerik (skala 1-10)
        "Academic Pressure": 5,
        "Work Pressure": 5,
        "Financial Stress": 5,
        "Study Satisfaction": 3,
        "Job Satisfaction": 3,
        "Work/Study Hours": 6,
        "CGPA": 7.5,
        "Age": 20,
        "Sleep Duration": 7.0,
        "Total_Pressure": 10,
        # Kategorikal
        "Dietary Habits": "Average",
        "Gender": "Male",
        "Profession": "Student",
        "Degree": "BSc",
        "Have you ever had suicidal thoughts ?": "No",
        "Family History of Mental Illness": "No",
        "Sleep_Quality": "Normal",
        "Financial Problem": "No",
        "Health Issue": "No",
        "Health Issues": "No",
    }

    for col in model_columns:
        if col not in data:
            data[col] = default_values.get(col, 0)
    
    df = pd.DataFrame([data])
    df = df[model_columns]
    logger.debug("Model input dtypes: %s", df.dtypes.to_dict())
    logger.debug("Model input row: %s", df.to_dict(orient="records")[0])
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

            # Penyesuaian ringan berbasis sinyal risiko agar lebih variatif namun tetap rasional
            try:
                ap = int(request.form["academic_pressure"])  # 1..10
                wp = int(request.form["work_pressure"])      # 1..10
                fs = int(request.form["financial_stress"])   # 1..10
                sd = float(request.form["sleep_duration"])   # jam
                diet = request.form.get("diet", "Average")
                st = request.form.get("suicidal_thoughts", "No")
                fh = request.form.get("family_history", "No")

                # Normalisasi sederhana ke 0..1
                pressure_signal = (ap + wp + fs) / 30.0  # rata-rata terukur tekanan
                # Pengaruh tidur: <5 buruk, 5-8 normal, >8 sedikit penalti
                if sd < 5:
                    sleep_effect = 0.25
                elif sd <= 8:
                    sleep_effect = 0.0
                else:
                    sleep_effect = -0.05

                diet_weight = {"Good": 0.0, "Average": 0.05, "Poor": 0.12}
                diet_effect = diet_weight.get(diet, 0.05)
                suicide_effect = 0.18 if st == "Yes" else 0.0
                family_effect = 0.08 if fh == "Yes" else 0.0

                heuristic_risk = pressure_signal * 0.65 + diet_effect + sleep_effect + suicide_effect + family_effect
                # clamp ke 0..1
                heuristic_risk = max(0.0, min(1.0, heuristic_risk))

                # Kombinasi cembung: mayoritas tetap dari model, sebagian kecil dari sinyal heuristik
                combined_prob = 0.85 * pred_prob + 0.15 * heuristic_risk
                combined_prob = _safe_clip(combined_prob)
                prediction = round(combined_prob * 100, 2)
                logger.debug(
                    "Pred raw=%.4f, heuristic=%.4f, combined=%.4f",
                    pred_prob,
                    heuristic_risk,
                    combined_prob,
                )
            except Exception as e:
                logger.warning("Heuristic adjustment failed, falling back to raw prob: %s", str(e))
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
