import os
import pickle
import requests
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

def download_file_from_github(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {filename} from GitHub")


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/janvi-png/fraud_flas/main/"

files_to_download = {
    "fraud_model.pkl": GITHUB_RAW_BASE + "fraud_model.pkl",
    "features.pkl": GITHUB_RAW_BASE + "features.pkl",
    "encoders.pkl": GITHUB_RAW_BASE + "encoders.pkl",
}

for filename, url in files_to_download.items():
    download_file_from_github(url, filename)

# Load the files
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

dropdown_fields = {}
for feature, encoder in encoders.items():
    dropdown_fields[feature] = list(encoder.classes_)

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

    return render_template("index.html", result=result, probability=probability,
                           features=feature_names, inputs=inputs, dropdowns=dropdown_fields)

if __name__ == '__main__':
    app.run(debug=True)
