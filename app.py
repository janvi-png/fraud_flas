import os
import requests
import pickle
import pandas as pd
from flask import Flask, render_template, request


def download_model_from_drive():
    MODEL_ID = "141CR3weueVcHV41Jf3SFnLWpzqLv8E41"
    MODEL_PATH = "fraud_model.pkl"
    if not os.path.exists(MODEL_PATH):
        print("Downloading fraud_model.pkl from Google Drive...")
        URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
        response = requests.get(URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

download_model_from_drive()

with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probability = None
    inputs = {}


    if request.method == 'POST':
        try:
            input_values = []
            for feature in feature_names:
                val = request.form.get(feature)
                inputs[feature] = val  
                if feature in encoders:
                    val = encoders[feature].transform([val])[0]
                else:
                    val = float(val)
                input_values.append(val)

            df_input = pd.DataFrame([input_values], columns=feature_names)
            pred = model.predict(df_input)[0]
            prob = model.predict_proba(df_input)[0][int(pred)]

            result = 'Fraudulent' if pred == 1 else 'Legitimate'
            probability = f"{prob * 100:.2f}%"
        except Exception as e:
            result = f"Error: {str(e)}"
            probability = "N/A"


    return render_template('index.html', result=result, probability=probability,
                           features=feature_names, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)
