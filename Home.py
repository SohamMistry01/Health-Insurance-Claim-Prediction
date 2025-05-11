import pandas as pd
import streamlit as st
import base64
import google.generativeai as genai
from dotenv import load_dotenv 
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

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


st.sidebar.title("Home")
page = st.sidebar.radio("Go to", ["About", "Dataset Info","Chat with Gemini"])

if page == "About":
    st.title("Health Insurance Claim Prediction System")
    st.divider()
    st.subheader("About Our Project: ")
    st.write("""The Health Insurance Claim Prediction System is a machine learning-based web application built using Streamlit, designed to estimate the annual medical premium cost for individuals. This project leverages the Random Forest algorithm to analyze key factors such as age, height, weight, and other relevant attributes to provide accurate cost predictions. By utilizing an intuitive interface, users can input their details and receive an estimated insurance premium, helping individuals and insurers make informed financial decisions. The model is trained on historical data to ensure reliable predictions, making it a valuable tool for policyholders and insurance providers alike.""")
    st.divider()
    st.subheader("Quick Links:")
    st.page_link("pages/Predict.py", label="⭕ Predict your Health Premium")
    st.page_link("pages/Model_Overview.py", label="⭕ Overview of Trained Model")
    st.page_link("pages/Visualizations.py", label="Explore Data Visualizations", icon="📊")
    st.divider()

elif page == "Dataset Info":
    st.title("🔎 Explore the Dataset") 
    st.divider()
    st.subheader("View the dataset:")
    df = pd.read_csv("Medicalpremium.csv")
    values_count = st.slider("Select the number of entries you want to display: ",0,986)
    values_displayed = df.head(values_count)
    st.write(values_displayed)
    st.divider()
    summary_stats = df.describe()
    st.subheader("Summary statistics of the above data set: ")
    st.write(summary_stats)
    st.divider()
    st.subheader("Correlation matrix of the dataset: ")
    st.write(df.corr())
    st.divider()

elif page == "Chat with Gemini":
    st.title("Your AI Assistant")
    st.divider()
    st.write("Chat with Gemini! Press 'Exit' to quit.")

    user_input = st.chat_input()
    if st.button("Exit"):
        st.caption("Chat dismissed...")
        
    if user_input:
        chat = model.start_chat()
        question = st.info(user_input)
        response = chat.send_message(user_input)
        message = st.chat_message(name='ai')
        message.success(response.text)
    
    



