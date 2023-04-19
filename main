import streamlit as st

st.title("Hydroponics Analyser")

mass = st.number_input("What is your mass(kg): ", step = 0.1)
height = st.number_input("What is your height(m): ", step = 0.01)

button = st.button("Calculate BMI")
if button:
  BMI = mass/height**2
  if BMI <= 18.4:
    st.success("You are Risk of Nutrional Deficiency.")
  elif BMI <= 22.9:
    st.success("You are Low Risk.")
  elif BMI <= 27.4:
    st.success("You are in Moderate Risk.")
  else:
    st.success("You are in High Risk.")
