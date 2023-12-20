import streamlit as st
import pickle
import numpy as np

# Load models
try:
    diabetes_model = pickle.load(
        open('D:\Streamlit\project\diabetes_models.sav', 'rb'))
    scaler = pickle.load(open('D:\Streamlit\project\scaler.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to make predictions


def predict_diabetes(inputs):
    # Use the same scaler to scale the input features
    inputs_scaled = scaler.transform([inputs])

    # Make predictions
    prediction = diabetes_model.predict(inputs_scaled)

    return prediction


st.title('Prediksi Diabetes')

# Input fields with default values
Pregnancies = st.number_input('Input nilai Pregnancies', value=0)
Glucose = st.number_input('Input nilai Glucose', value=0)
BloodPressure = st.number_input('Input nilai Blood Pressure', value=0)
SkinThickness = st.number_input('Input nilai Skin Thickness', value=0)
Insulin = st.number_input('Input nilai Insulin', value=0)
BMI = st.number_input('Input nilai BMI', value=0.0)
DiabetesPedigreeFunction = st.number_input(
    'Input nilai Diabetes Pedigree Function', value=0.0, format="%0.3f")
Age = st.number_input('Input nilai Age', value=0)

diagnosis = ''

if st.button('Tes Prediksi Diabetes'):
    # Validate inputs
    inputs = np.array([Pregnancies, Glucose, BloodPressure,
                      SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.write("Input values:", inputs)  # Debug statement

    if not all(np.isfinite(inputs)):
        st.error("Please enter valid numeric values for all input fields.")
        st.stop()

    st.write("Model weights:", diabetes_model.coef_)  # Debug statement
    st.write("Model intercept:", diabetes_model.intercept_)  # Debug statement

    prediction = predict_diabetes(inputs)

    st.write("Model prediction:", prediction)  # Debug statement

    if prediction[0] == 0:
        diagnosis = 'Pasien Tidak Terkena Diabetes'
    else:
        diagnosis = "Pasien Terkena Diabetes"

    st.success(diagnosis)
