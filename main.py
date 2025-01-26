import streamlit as st
import pickle
import numpy as np

# Load the trained SVM model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function for prediction
def make_prediction(input_data):
    try:
        # Convert user input to a numpy array and reshape it to match the model's input format
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction[0]
    except Exception as e:
        return f"Error in prediction: {e}"

# Streamlit application
st.title("SVM Model Prediction App")
st.write("This web application allows users to input data and get predictions from an SVM model trained on assignment 3 data.")

# User input form
st.header("Enter Input Data")
st.write("Please provide the required input features for prediction.")

# Create input fields for the dataset features
gender = st.selectbox("Gender", ["Male", "Female"])
# Encode Gender as 0 for Male and 1 for Female
gender_encoded = 0 if gender == "Male" else 1

age = st.number_input("Age", min_value=0, max_value=100, value=25)
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)

# Collect input data
input_data = [gender_encoded, age, estimated_salary]

# Prediction button
if st.button("Predict!"):
    prediction = make_prediction(input_data)
    st.header("Prediction Result")
    if prediction == 1:
        st.write("The prediction is: **Purchased**")
    else:
        st.write("The prediction is: **Not Purchased**")
