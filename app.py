from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load model and files
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
