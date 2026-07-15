import streamlit as st
import pickle
import numpy as np

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.goodfon.com/wallpaper/big/c/5c/vino-butylka-bokaly-krasnoe.webp");
        background-size: cover;
}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
with open("model_RF.pkl", "rb") as f:
    model = pickle.load(f)


# Load scaler if used
try:
    with open("scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None
st.title("Wine Quality Prediction")
st.image("image.jpg",use_container_width=True)
st.write("Enter the details of the wine to predict its quality.")

#user inputs
age = st.number_input("Fixed Acidity")
sex = st.number_input("Volatile Acidity")
cp = st.number_input("Citric Acid")
trestbps = st.number_input("Residual Sugar")
chol = st.number_input("Chlorides")
fbs = st.number_input("Free Sulfur Dioxide")
restecg = st.number_input("Total Sulfur Dioxide")
thalach = st.number_input("Density")
oldpeak = st.number_input("pH")
slope = st.number_input("Sulphates")    
ca = st.number_input("Alcohol")


# Make predictions
if st.button("Predict"):
    if scaler:
        inputs = scaler.transform(np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, oldpeak, slope, ca]]))
    else:
        inputs = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, oldpeak, slope, ca]])
    prediction = model.predict(inputs)
    st.write(f"The predicted quality of the wine is: {prediction[0]}")