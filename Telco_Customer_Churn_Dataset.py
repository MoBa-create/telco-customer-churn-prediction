import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['Churn'])
y = df['Churn']
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

sample_customer = X_test.iloc[[0]]
actual_status = y_test.iloc[0]

prediction = rf_model.predict(sample_customer)[0]
probability = rf_model.predict_proba(sample_customer)[0]

joblib.dump(rf_model, "churn_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
print("The model and columns have been successfully trained and saved!")