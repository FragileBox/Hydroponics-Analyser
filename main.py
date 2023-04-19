import streamlit as st

st.title("Hydroponics Analyser")

mass = st.number("What is your mass(kg): ", step = 100)
height = st.slider("What is your height(m): ", step = 100)

button = st.button("Predict Conditions")
if button:
    st.success("Analysing data...")
