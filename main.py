import streamlit as st
from streamlit_extras.switch_page_button import *

st.title("Hydroponics Analyser")

new_sol = st.number_input("Nutrient solution to be added(kg): ", step = 0.01)
add_sol = st.number_input("Nutrient solution already added(kg): ", step = 0.01)
Sodium = st.slider("Na content(mg): ", step = 1, min_value = 400, max_value = 2100)
Potassium = st.slider("K content(mg): ", step = 1, min_value = 400, max_value = 2100)
Magnessium = st.slider("Mg content(mg): ", step = 1, min_value = 400, max_value = 2100)
Calcium = st.slider("Ca content(mg): ", step = 1, min_value = 400, max_value = 2100)

button = st.button("Predict Conditions")
if button:
    switch_page("Result")
