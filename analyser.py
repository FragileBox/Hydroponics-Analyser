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
model = ensemble.RandomForestRegressor(random_state=2020)

# Split training into some for training and some for testing
Xtrain_S, Xtest_S, ytrain_S, ytest_S = model_selection.train_test_split(X_S, y_S, test_size=0.2, random_state=10)
Xtrain_F, Xtest_F, ytrain_F, ytest_F = model_selection.train_test_split(X_F, y_F, test_size=0.2, random_state=10)

# Perform regression on the data
model.fit(X_S, y_S)
model.score(Xtest_S, ytest_S)
model.fit(X_F, y_F)
model.score(Xtest_F, ytest_F)

# Make Predictions
for features in XforPredictions:
  yPredictions = model.predict(XforPredictions)
  st.success(f"Predictions for {features}: {yPredictions}")
