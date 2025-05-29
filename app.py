from flask import Flask, render_template, request
import pandas as pd
from catboost import CatBoostClassifier
import joblib

app = Flask(__name__)

# Load model and columns
model = CatBoostClassifier()
model.load_model("catboost_depression_model.cbm")
model_columns = joblib.load("model_columns.pkl")  # pastikan file ini ada

def preprocess_input(form):
    data = {
        "Gender": form["gender"].capitalize(),
        "Age": int(form["age"]),
        "CGPA": float(form["cgpa"]),
        "Sleep Duration": float(form["sleep_duration"]),
        "Academic Pressure": int(form["academic_pressure"]),
        "Work Pressure": int(form["work_pressure"]),
        "Financial Stress": int(form["financial_stress"]),
        "Family History of Mental Illness": form["family_history"].capitalize(),
        "Financial Problem": form["financial_problem"].capitalize(),
        "Health Issue": form["health_issue"].capitalize(),
        "Social Support": form["social_support"].capitalize(),
        "Diet": form["diet"].capitalize(),
        "Lost Interest": form["lost_interest"].capitalize(),
        "Have you ever had suicidal thoughts ?": form["suicidal_thoughts"].capitalize()
    }

    # Fitur turunan
    data["Total_Pressure"] = data["Academic Pressure"] + data["Work Pressure"]

    if data["Sleep Duration"] < 5:
        data["Sleep_Quality"] = "Poor"
    elif 5 <= data["Sleep Duration"] <= 8:
        data["Sleep_Quality"] = "Normal"
    else:
        data["Sleep_Quality"] = "Over"

    # Konversi CGPA 0-4 ke 0-10
    data["CGPA"] = (data["CGPA"] / 4) * 10

    df = pd.DataFrame([data])

    # Tambahkan kolom yang mungkin tidak ada
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_df = preprocess_input(request.form)
        pred_prob = model.predict_proba(input_df)[0][1]
        prediction = round(pred_prob * 100, 2)



    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
