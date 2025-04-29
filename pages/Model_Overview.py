import streamlit as st
import base64
import time

with st.spinner("Running the model..."):
    time.sleep(1)
with st.spinner("Loading the notebook..."):
    time.sleep(1)

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

st.title("Model Building Overview")

st.divider()

with open("pages\model2.html", "r", encoding="utf-8") as f:
    html_content = f.read()

st.components.v1.html(html_content, height=800, scrolling=True)
