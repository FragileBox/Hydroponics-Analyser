import streamlit as st
import pandas as pd

# Load CSVs
data_solution = pd.read_csv('Nutrient_solution_consumption.csv')
data_fruit = pd.read_csv('Fruit_cation.csv')

# Drops missing values
for data in [data_solution, data_fruit]:
  data = data.dropna(axis=0)

# Convert strings to numbers
types = {"Fruits (from harvest)": 1}
data_fruit.Organ_harvested = [types[x] for x in data_fruit.Organ_harvested]

# Removing outliers
# We are not removing outliers in this case since the sample size is too small
if False:
  for x in ["TotalVolume"]:
      q3, q1 = np.percentile(avo_TRAIN[x], [75, 25])
      IQR = q3 - q1
  
      upper_bound = q3 + 1.0 * IQR
      lower_bound = q1 - 1.0 * IQR
      avo_TRAIN[x] = avo_TRAIN[x][avo_TRAIN[x] < upper_bound]
      avo_TRAIN[x] = avo_TRAIN[x][avo_TRAIN[x] > lower_bound]
  avo_TRAIN = avo_TRAIN.dropna(axis=0)
  
# Select data for learning
features_solution = ["EC_limit", "NS_new", "NS_added"]
features_fruit = ["Na", "K", "Mg", "Ca"]

X_S = data_solution[features_solution]
y_S = data_solution.NS_residual
X_F = data_fruit[features_fruit]
y_F = data_fruit.Organ_harvested

from sklearn import *

# Pick the regression model we want to use
model_S = ensemble.RandomForestRegressor(random_state=2020)
model_F = ensemble.RandomForestRegressor(random_state=2020)

# Split training into some for training and some for testing
Xtrain_S, Xtest_S, ytrain_S, ytest_S = model_selection.train_test_split(X_S, y_S, test_size=0.2, random_state=10)
Xtrain_F, Xtest_F, ytrain_F, ytest_F = model_selection.train_test_split(X_F, y_F, test_size=0.2, random_state=10)

# Perform regression on the data
model_S.fit(X_S, y_S)
model_F.fit(X_F, y_F)

st.title("Hydroponics Analyser")

ec_limit = st.number_input("EC Limit: ", step = 0.1)
new_sol = st.number_input("Nutrient solution to be added(kg): ", step = 0.01)
add_sol = st.number_input("Nutrient solution already added(kg): ", step = 0.01)
Sodium = st.slider("Na content(mg): ", step = 1, min_value = 400, max_value = 2100)
Potassium = st.slider("K content(mg): ", step = 1, min_value = 400, max_value = 2100)
Magnesium = st.slider("Mg content(mg): ", step = 1, min_value = 400, max_value = 2100)
Calcium = st.slider("Ca content(mg): ", step = 1, min_value = 400, max_value = 2100)

button = st.button("Predict Conditions")
if button:
    Predictions_S = model_S.predict([[ec_limit, new_sol, add_sol]])
    Predictions_F = model_F.predict([[Sodium, Potassium, Magnesium, Calcium]])
    normalized_S = (Predictions_S-0.0)/(max(7.6)-min(0.0))*(3/4)
    st.success(f"Probability of success: {normalized_S*(Predictions_F/2)}%")
