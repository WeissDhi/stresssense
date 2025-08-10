import pandas as pd
from catboost import CatBoostClassifier
import joblib
import numpy as np

# Load model and columns
model = CatBoostClassifier()
model.load_model("catboost_depression_model.cbm")
model_columns = joblib.load("model_columns.pkl")

print("Model columns:", model_columns)
print("Number of columns:", len(model_columns))

# Test dengan beberapa input yang berbeda
test_cases = [
    {
        "name": "Case 1 - testt1",
        "data": {
            "Gender": "Male",
            "Age": 22,
            "CGPA": 3.8,
            "Sleep Duration": 6,
            "Academic Pressure": 10,
            "Work Pressure": 1,
            "Financial Stress": 10,
            "Family History of Mental Illness": "No",
            "Financial Problem": "Yes",
            "Health Issue": "No",
            "Diet": "Poor",
            "Have you ever had suicidal thoughts ?": "No"
        }
    },
    {
        "name": "Case 2 - High Risk",
        "data": {
            "Gender": "Female",
            "Age": 22,
            "CGPA": 2.5,
            "Sleep Duration": 4.0,
            "Academic Pressure": 9,
            "Work Pressure": 8,
            "Financial Stress": 9,
            "Family History of Mental Illness": "Yes",
            "Financial Problem": "Yes",
            "Health Issue": "Yes",
            "Diet": "Poor",
            "Have you ever had suicidal thoughts ?": "Yes"
        }
    }
]

def preprocess_input(form_data):
    data = {
        "id": 0,
        "Gender": form_data["Gender"],
        "Age": int(form_data["Age"]),
        "Profession": "Student",
        "Academic Pressure": int(form_data["Academic Pressure"]),
        "Work Pressure": int(form_data["Work Pressure"]),
        "CGPA": (float(form_data["CGPA"]) / 4) * 10,
        "Study Satisfaction": 3,
        "Job Satisfaction": 0,
        "Sleep Duration": float(form_data["Sleep Duration"]),
        "Dietary Habits": form_data["Diet"],
        "Degree": "BSc",
        "Have you ever had suicidal thoughts ?": form_data["Have you ever had suicidal thoughts ?"],
        "Work/Study Hours": 5,
        "Financial Stress": int(form_data["Financial Stress"]),
        "Family History of Mental Illness": form_data["Family History of Mental Illness"],
        "Total_Pressure": int(form_data["Academic Pressure"]) + int(form_data["Work Pressure"]),
        "Sleep_Quality": "Normal"
    }

    # Sleep Quality calculation
    if data["Sleep Duration"] < 5:
        data["Sleep_Quality"] = "Poor"
    elif 5 <= data["Sleep Duration"] <= 8:
        data["Sleep_Quality"] = "Normal"
    else:
        data["Sleep_Quality"] = "Over"

    df = pd.DataFrame([data])
    
    # Tambahkan kolom yang mungkin tidak ada
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match model_columns
    df = df[model_columns]
    
    return df

print("\n" + "="*50)
print("TESTING MODEL PREDICTIONS")
print("="*50)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{test_case['name']}:")
    print("-" * 30)
    
    try:
        input_df = preprocess_input(test_case['data'])
        print("Input data shape:", input_df.shape)
        
        # Print only key columns for clarity
        key_cols = ['Gender', 'Age', 'CGPA', 'Sleep Duration', 'Academic Pressure', 
                   'Work Pressure', 'Financial Stress', 'Family History of Mental Illness',
                   'Have you ever had suicidal thoughts ?', 'Total_Pressure', 'Sleep_Quality']
        
        print("Key input values:")
        for col in key_cols:
            if col in input_df.columns:
                print(f"  {col}: {input_df[col].iloc[0]}")
        
        pred_prob = model.predict_proba(input_df)[0]
        prediction = round(pred_prob[1] * 100, 2)
        
        print(f"\nPrediction probabilities: {pred_prob}")
        print(f"Risk percentage: {prediction}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("MODEL ANALYSIS")
print("="*50)

# Check model features
print(f"Model feature count: {len(model.feature_names_)}")
print(f"Model tree count: {model.tree_count_}")
print("Model feature names:", model.feature_names_)

# Test with random data
print("\n" + "="*50)
print("RANDOM DATA TEST")
print("="*50)

random_data = {
    "Gender": "Male",
    "Age": 21,
    "CGPA": 3.0,
    "Sleep Duration": 7.0,
    "Academic Pressure": 5,
    "Work Pressure": 4,
    "Financial Stress": 5,
    "Family History of Mental Illness": "No",
    "Financial Problem": "No",
    "Health Issue": "No",
    "Diet": "Average",
    "Have you ever had suicidal thoughts ?": "No"
}

try:
    input_df = preprocess_input(random_data)
    pred_prob = model.predict_proba(input_df)[0]
    prediction = round(pred_prob[1] * 100, 2)
    
    print(f"Random data prediction: {prediction}%")
    print(f"Probabilities: {pred_prob}")
    
except Exception as e:
    print(f"Error with random data: {e}")
    import traceback
    traceback.print_exc() 