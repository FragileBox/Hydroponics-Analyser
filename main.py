import streamlit as st

st.title("Hydroponics Analyser")

new_sol = st.number_input("Nutrient solution to be added(kg): ", step = 0.01)
add_sol = st.number_input("Nutrient solution already added(kg): ", step = 0.01)
Sodium = st.slider("Na content(mg): ", step = 1, min_value = 400, max_value = 2100)

button = st.button("Predict Conditions")
if button:
    st.success("Analysing data...")
