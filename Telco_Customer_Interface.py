import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🛡️ نظام التنبؤ بمغادرة عملاء شركة الاتصالات")
st.write("أدخل بيانات العميل لمعرفة احتمالية مغادرته للخدمة:")

monthly_charges = st.number_input("المبلغ الشهري (Monthly Charges)", value=70.0)
total_charges = st.number_input("إجمالي المبالغ المدفوعة (TotalCharges)", value=500.0)
tenure = st.slider("عدد أشهر الاشتراك (Tenure)", 0, 72, 12)

contract = st.selectbox("نوع العقد (Contract)", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("طريقة الدفع (Payment Method)", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

if st.button("تنبؤ حالة العميل"):
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["tenure"] = tenure

    contract_col = f"Contract_{contract}"
    if contract_col in input_data.columns:
        input_data[contract_col] = 1
        
    payment_col = f"PaymentMethod_{payment_method}"
    if payment_col in input_data.columns:
        input_data[payment_col] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ تحذير: هذا العميل معرض لمغادرة الشركة بنسبة احتمالية: {probability * 100:.2f}%")
    else:
        st.success(f"✅ العميل مستقر ومستمر مع الشركة (احتمالية المغادرة: {probability * 100:.2f}%)")
