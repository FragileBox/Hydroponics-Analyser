import streamlit as st
import pandas as pd
import numpy as np
from sklearn import *

# Just to add background for app
def add_bg_from_url(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: linear-gradient(rgba(0, 0, 0, 0.60), rgba(0, 0, 0, 0.60)), url({url}), no-repeat;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Main Application
st.title("Hydroponics Analyser")
option = st.selectbox("Select type of vegetable", ["Tomato","Eggplant"])

if option == "Tomato":
  add_bg_from_url("https://economictimes.indiatimes.com/thumb/msid-95423731,width-1200,height-900,resizemode-4,imgsize-56196/tomatoes-canva.jpg?from=mdr")
  ec_limit = st.number_input("EC Limit: ", step = 0.1)
  new_sol = st.number_input("Nutrient solution to be added(kg): ", step = 0.01)
  add_sol = st.number_input("Nutrient solution already added(kg): ", step = 0.01)
  Sodium = st.slider("Na content(mg/L): ", step = 1, min_value = 400, max_value = 2100)
  Potassium = st.slider("K content(mg/L): ", step = 1, min_value = 24000, max_value = 32000)
  Magnesium = st.slider("Mg content(mg/L): ", step = 1, min_value = 600, max_value = 3000)
  Calcium = st.slider("Ca content(mg/L): ", step = 1, min_value = 300, max_value = 3500)

  button = st.button("Predict Conditions")
  if button:
      # Tomato model
      # Load CSVs
      data_solution = pd.read_csv('Nutrient_solution_consumption.csv')
      data_fruit = pd.read_csv('Fruit_cation.csv')

      # Drops missing values
      for data in [data_solution, data_fruit]:
        data = data.dropna(axis=0)

      # Convert strings to numbers
      types = {"No Fruits (from harvest)": 0,"Fruits (from harvest)": 1}
      data_fruit.Organ_harvested = [types[x] for x in data_fruit.Organ_harvested]

      # Removing outliers
      # We are not removing outliers in this case since the sample size is too small
      if False:
        for x in []: #To add Features
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

      # Pick the regression model we want to use
      model_T_S = ensemble.RandomForestRegressor(random_state=2020)
      model_T_F = ensemble.RandomForestRegressor(random_state=2020)

      # Split training into some for training and some for testing
      Xtrain_S, Xtest_S, ytrain_S, ytest_S = model_selection.train_test_split(X_S, y_S, test_size=0.2, random_state=10)
      Xtrain_F, Xtest_F, ytrain_F, ytest_F = model_selection.train_test_split(X_F, y_F, test_size=0.2, random_state=10)

      # Perform regression on the data
      model_T_S.fit(X_S, y_S)
      model_T_F.fit(X_F, y_F)
      
      # Make Predictions
      Predictions_S = model_T_S.predict([[ec_limit, new_sol, add_sol]])
      Predictions_F = model_T_F.predict([[Sodium, Potassium, Magnesium, Calcium]])
      normalized_S = ((Predictions_S-0.0)/(7.2-0.0))/2
      st.success(f"Probability of success: {'%.2f'%((normalized_S+(Predictions_F/2))*100)}%")

elif option == "Eggplant":
  add_bg_from_url("https://cdn-prod.medicalnewstoday.com/content/images/articles/279/279359/two-eggplants-on-a-wooden-table.jpg")
  ec_limit = st.number_input("EC Limit: ", step = 0.1)
  #pH = st.number_input("pH: ", step = 0.1)
  #Temp = st.slider("Temperature(Farenheit): ", step = 1, max_value = 100)
  #Humidity = st.slider("Humidity(%): ", step = 0.1, min_value = 0.0, max_value = 100.0)
  Nitrogen = st.slider("N content(%): ", step = 0.1, min_value = 0.0, max_value = 100.0)
  Phosporous = st.slider("P content(%): ", step = 0.1, min_value = 0.0, max_value = 100.0)
  Potassium = st.slider("K content(%): ", step = 0.1, min_value = 0.0, max_value = 100.0)

  button = st.button("Predict Conditions")
  if button:
      # Eggplant model
      # Load CSVs
      data_nut = pd.read_csv('Eggplant_Nutrient.csv')
      #data_env = pd.read_csv('Eggplant_Environment.csv')

      # Removing leading/trailing spaces
      data_nut = data_nut.rename(columns={' Fruit': 'Fruit'})
      data_nut = data_nut.rename(columns={' Phosporus(%)': 'Phosporus(%)'})
      data_nut = data_nut.rename(columns={' Potassium(%)': 'Potassium(%)'})

      # Drops missing values
      for data in [data_nut]:
        data = data.dropna(axis=0)

      # Convert strings to numbers
      types = {"No Fruits (from harvest)": 0,"Fruits (from harvest)": 1}
      for data in [data_nut]:
        data.Fruit = [types[x] for x in data.Fruit]
      
      # Setting Variance as Y axis
      data_nut["Norm_Nitrogen"] = np.where(data_nut["Nitrogen(%)"] <= 10, (data_nut["Nitrogen(%)"]-8.5)/(10-8.5), (11.5-data_nut["Nitrogen(%)"])/(11.5-10))
      data_nut["Norm_Phosporus"] = np.where(data_nut["Phosporus(%)"] <= 10, (data_nut["Phosporus(%)"]-8.5)/(10-8.5), (11.5-data_nut["Phosporus(%)"])/(11.5-10))
      data_nut["Norm_Potassium"] = np.where(data_nut["Potassium(%)"] <= 10, (data_nut["Potassium(%)"]-8.5)/(10-8.5), (11.5-data_nut["Potassium(%)"])/(11.5-10))
      data_nut["Total_Var"] = (data_nut["Norm_Nitrogen"]+data_nut["Norm_Phosporus"]+data_nut["Norm_Potassium"])/3
      data_nut.drop(["Norm_Nitrogen", "Norm_Phosporus", "Norm_Potassium"], axis='columns', inplace=True)
      data_nut = data_nut[data_nut["Fruit"] != 1]

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
      features_nut = ["Nitrogen(%)", "Phosporus(%)", "Potassium(%)"]
      features_env = ["pH", "Temperature(Farenheit)", "Humidity"]

      X_N = data_nut[features_nut]
      y_N = data_nut.Total_Var
      #X_E = data_env[features_env]
      #y_E = data_env.Fruit

      # Pick the regression model we want to use
      model_E_N = ensemble.RandomForestRegressor(random_state=2020)
      #model_E_E = ensemble.RandomForestRegressor(random_state=2020)

      # Split training into some for training and some for testing
      Xtrain_N, Xtest_N, ytrain_N, ytest_N = model_selection.train_test_split(X_N, y_N, test_size=0.2, random_state=10)
      #Xtrain_E, Xtest_E, ytrain_E, ytest_E = model_selection.train_test_split(X_E, y_E, test_size=0.2, random_state=10)

      # Perform regression on the data
      model_E_N.fit(X_N, y_N)
      #model_E_E.fit(X_E, y_E)
      
      # Make Predictions
      #Predictions_E = model_E_E.predict([[pH, Temp, Humidity]])
      Predictions_N = model_E_N.predict([[Nitrogen, Phosporous, Potassium]])
      st.success(f"Probability of success: {'%.2f'%((Predictions_N)*100)}%")
