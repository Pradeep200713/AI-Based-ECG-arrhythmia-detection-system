import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))

st.title("💓 ECG Arrhythmia Detection System")

st.write("Enter ECG values:")

# Example: 5 inputs (you can change based on dataset)
input1 = st.number_input("Value 1")
input2 = st.number_input("Value 2")
input3 = st.number_input("Value 3")
input4 = st.number_input("Value 4")
input5 = st.number_input("Value 5")

if st.button("Predict"):
    data = np.array([[input1, input2, input3, input4, input5]])
    result = model.predict(data)

    if result[0] == 0:
        st.success("Normal ❤️")
    else:
        st.error("Abnormal ⚠️")
