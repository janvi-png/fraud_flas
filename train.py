
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

csv_url = "https://drive.google.com/uc?export=download&id=12dX71hdB6qNeuK3CJ3VxPJtLLMckKsFJ"
df = pd.read_csv(csv_url)
df = df.dropna()


selected_features = ['Transaction_Amount', 'Age', 'Account_Balance', 'Account_Type', 'City', 'Gender']
target_col = 'Is_Fraud'


encoders = {}
for col in selected_features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

X = df[selected_features]
y = df[target_col]


with open('features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 9. Print accuracy
print(f"Model trained. Accuracy: {model.score(X_test, y_test):.2f}")
