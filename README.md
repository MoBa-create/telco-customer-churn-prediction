# Telco Customer Churn Prediction & Web App

A complete Machine Learning end-to-end project that predicts whether a telecommunications customer is likely to churn (leave the company) or stay, featuring an interactive web application built with Streamlit.

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project uses the popular IBM Telco Customer Churn dataset to train a supervised machine learning model capable of predicting customer retention based on key service usage features, contract types, and payment methods. The trained model is then deployed locally via an interactive web interface.

---

## 🚀 Key Features
- **Data Preprocessing & Cleaning:** Handles missing values in total charges, encodes categorical variables using one-hot encoding (`pd.get_dummies`), and maps target labels.
- **Machine Learning Model:** Utilizes a **Random Forest Classifier** (`RandomForestClassifier`) with balanced class weights to effectively handle imbalanced datasets.
- **Model Persistence:** Saves the trained model and dataset column structures using `joblib` (`churn_model.pkl`, `model_columns.pkl`) to ensure consistent inference without retraining.
- **Interactive Web UI:** Built using **Streamlit** to allow users to dynamically input customer details (Monthly Charges, Total Charges, Tenure, Contract, Payment Method) and instantly get real-time churn predictions with probability scores.

---

## 🛠️ Tech Stack & Libraries
- **Python 3.x**
- **Pandas & NumPy** (Data Manipulation & Preprocessing)
- **Scikit-Learn** (Machine Learning & Evaluation)
- **Joblib** (Model Saving & Loading)
- **Streamlit** (Web Application Interface)

---

## 📁 Project Structure
```text
├── train_model.py         # Script to download data, train the Random Forest model, and save pkl files
├── app.py                 # Streamlit web application interface for user predictions
├── churn_model.pkl        # Serialized trained Random Forest model
├── model_columns.pkl      # Serialized feature columns list for input alignment
└── requirements.txt       # List of required Python packages

⚙️ Installation & Running Locally

1 . Clone the repository:
	git clone https://github.com/MoBa-create/E-Commerce-Customer-Segmentation-Hierarchical.git
	cd YOUR_REPOSITORY_NAME

2 . Install the required dependencies:
	pip install -r requirements.txt

3 . Train the model (if pkl files are missing):
	python train_model.py

4 . Run the Streamlit web app:
	streamlit run app.py

📊 Dataset Source
The model is trained on the publicly available IBM Telco Customer Churn dataset, hosted via GitHub.
