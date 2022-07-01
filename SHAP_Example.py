import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt


df = pd.read_csv('wine.csv', delimiter=';') # Load the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
# The target variable is 'quality'.
Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Build the model with the random forest regression algorithm:
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)

import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
# shap.summary_plot(shap_values, X_train, plot_type="bar")
#
# shap.summary_plot(shap_values, X_train)
#
# import matplotlib.pyplot as plt
#
# shap.dependence_plot("alcohol", shap_values, X_train)

new_df = pd.DataFrame(shap_values)

shap.force_plot(new_df.expected_value, new_df)
# shap.force_plot(shap_values.expected_value[0], shap_values[0])




# shap.plots.bar(new_df.cohorts(2).abs.mean(0))



shap.plots.waterfall(shap_values[0])