#1. import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('C:/Users/janvi/OneDrive/Desktop/DMA_LAB/fraud_flas/Bank_Transaction_Fraud_Detection.csv')  
df = df.dropna()

# Select  features i.e X & Y
selected_features = ['Transaction_Amount', 'Age', 'Account_Balance', 'Account_Type', 'City','Gender']
target_col = 'Is_Fraud'

# Label ENCODING 
encoders = {}
for col in selected_features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

X = df[selected_features]
y = df[target_col]


# Save the order of features
with open('features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Train-test split I.E 80/20 TRAINING/TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model I.E RANDOM FOREST CLASSIFICAtion
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)


print(f"Model trained. Accuracy: {model.score(X_test, y_test):.2f}")
