import os
import requests
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


def download_model_from_drive():
    MODEL_ID = "141CR3weueVcHV41Jf3SFnLWpzqLv8E41"
    MODEL_PATH = "fraud_model.pkl"
    URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

    if os.path.exists(MODEL_PATH):
        return

    print("Downloading model from Google Drive...")

    session = requests.Session()
    response = session.get(URL, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': MODEL_ID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("Model download complete.")


download_model_from_drive()


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probability = None
    inputs = {}

  
    try:
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('features.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
    except Exception as e:
        return f"Critical Error: {e}"

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
