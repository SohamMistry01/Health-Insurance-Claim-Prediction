import streamlit as st
import pandas as pd
import joblib
import base64

# Background Image CSS
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    bg_image_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Set Background
set_bg("bg_2.jpg")

# Load the trained model
@st.cache_resource()
def load_model():
    model = joblib.load("pages\\new_rf_model.pkl")  # Save and load your model here
    return model

# Load your model
model = load_model()


# Title of the app
st.title("Predict Your Medical Premium")
st.divider()

# Input fields for user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
st.divider()
diabetes = st.checkbox("Diabetes")
bp_problems = st.checkbox("Blood Pressure Problems")
transplants = st.checkbox("Any Transplants")
chronic_diseases = st.checkbox("Any Chronic Diseases")
st.divider()
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
bmi = weight / (height/100)**2
st.info(f"Your BMI is: {bmi}")
st.divider()
allergies = st.checkbox("Known Allergies")
cancer_history = st.checkbox("History of Cancer in Family")
st.divider()
major_surgeries = st.slider("Number of Major Surgeries", 0, 5, 1)
st.divider()

# Convert categorical inputs to numeric
diabetes = 1 if diabetes else 0
bp_problems = 1 if bp_problems else 0
transplants = 1 if transplants else 0
chronic_diseases = 1 if chronic_diseases else 0
allergies = 1 if allergies else 0
cancer_history = 1 if cancer_history else 0

# Create input DataFrame
data = pd.DataFrame (
    {
    'Age': [age],
    'Diabetes': [diabetes],
    'BloodPressureProblems': [bp_problems],
    'AnyTransplants': [transplants],
    'AnyChronicDiseases': [chronic_diseases],
    'Height': [height],
    'Weight': [weight],
    'KnownAllergies': [allergies],
    'HistoryOfCancerInFamily': [cancer_history],
    'NumberOfMajorSurgeries': [major_surgeries],
    'BMI': [bmi]
    }
)

expected_features = [
    'Age', 'Diabetes', 'BloodPressureProblems', 'AnyTransplants',
 'AnyChronicDiseases', 'Height', 'Weight','KnownAllergies',
 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries', 'BMI'
]

# ✅ Keep only the required features
data = data[expected_features]

st.write("Select the medical insurance cover you want : ")
options = st.selectbox("Select from the dropdown list: ",
                       ("5 Lakhs", "10 Lakhs", "20 Lakhs", "50 Lakhs", "1 Crore"), 
                       index=None, placeholder="Select the amount")
if options is None:
    st.warning("Please select an insurance cover to proceed.")
    st.stop()
st.write("Your Selected medical insurance is: ",options)
st.divider()
st.write("Review your input data:", data)
st.divider()

# Prediction
if st.button('Predict Premium Price'):
    predicted_premium = model.predict(data)
    if options=="5 Lakhs":
        final_premium = predicted_premium*0.4
        st.success(f"The predicted health insurance premium price is: INR {final_premium.flatten()[0]:.2f} per annum")
    elif options=="10 Lakhs":
        final_premium = predicted_premium*0.5
        st.success(f"The predicted health insurance premium price is: INR {final_premium.flatten()[0]:.2f} per annum")
    elif options=="20 Lakhs":
        final_premium = predicted_premium*0.6
        st.success(f"The predicted health insurance premium price is: INR {final_premium.flatten()[0]:.2f} per annum")
    elif options=="50 Lakhs":
        final_premium = predicted_premium*0.75
        st.success(f"The predicted health insurance premium price is: INR {final_premium.flatten()[0]:.2f} per annum")
    else:
        st.success(f"The predicted health insurance premium price is: INR {predicted_premium.flatten()[0]:.2f} per annum")
st.divider()

if st.button('Check BMI status'):
    if bmi < 16:
        category = "Severely Underweight"
    elif 16 <= bmi < 17:
        category = "Very Underweight"
    elif 17 <= bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal (Healthy Weight)"
    elif 25 <= bmi < 30:
        category = "Overweight"
    elif 30 <= bmi < 35:
        category = "Obesity Class I (Moderate)"
    elif 35 <= bmi < 40:
        category = "Obesity Class II (Severe)"
    else:
        category = "Obesity Class III (Morbid)"

    st.info(f"Category: **{category}**")

st.divider()
st.subheader("BMI Classification Table")

bmi_data = {
    "BMI Range (kg/m²)": [
        "Below 16.0",
        "16.0 – 16.9",
        "17.0 – 18.4",
        "18.5 – 24.9",
        "25.0 – 29.9",
        "30.0 – 34.9",
        "35.0 – 39.9",
        "40.0 and above"
    ],
    "Category": [
        "Severely Underweight",
        "Very Underweight",
        "Underweight",
        "Normal (Healthy Weight)",
        "Overweight",
        "Obesity Class I (Moderate)",
        "Obesity Class II (Severe)",
        "Obesity Class III (Morbid)"
    ]
}

bmi_df = pd.DataFrame(bmi_data)
st.table(bmi_df)