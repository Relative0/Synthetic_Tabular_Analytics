import shap
shap.initjs()
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

df = pd.read_csv('wine.csv', sep=';')

df.columns
df['quality'] = df['quality'].astype(int)
df.head()
df['quality'].hist()

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

X_test.shape

X_test.mean()

X_test.iloc[10,:]

#################
# Random Forest #
#################
rf = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
rf.fit(X_train, Y_train)
print(rf.feature_importances_)

importances = rf.feature_importances_
indices = np.argsort(importances)

features = X_train.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

X_test[0:1]

################
#   The SHAP   #
################
import shap
shap.initjs()
rf_explainer = shap.KernelExplainer(rf.predict, X_test)
rf_shap_values = rf_explainer.shap_values(X_test)

# X_test
#
# rf_explainer(X_test)
#
# rf_shap_values
#
# rf_explainer.expected_value

# plot the SHAP values for the 10th observation
shap.force_plot(rf_explainer.expected_value, rf_shap_values[10,:], X_test.iloc[10,:]) #, link="logit")

shap.force_plot(rf_explainer.expected_value, rf_shap_values, X_test)

shap.summary_plot(rf_shap_values, X_test, plot_type="bar")

shap.summary_plot(rf_shap_values, X_test)

shap.dependence_plot("alcohol", rf_shap_values, X_test)

##############
#    GBM     #
##############
from sklearn import ensemble
n_estimators = 500
gbm = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=0)
gbm.fit(X_train, Y_train)

gbm_explainer = shap.KernelExplainer(gbm.predict, X_test)
gbm_shap_values = gbm_explainer.shap_values(X_test)


shap.summary_plot(gbm_shap_values, X_test)

shap.dependence_plot("alcohol", gbm_shap_values, X_test)

# plot the SHAP values for the 10th observation
shap.force_plot(gbm_explainer.expected_value,gbm_shap_values[10,:], X_test.iloc[10,:]) #, link="logit")

shap.force_plot(gbm_explainer.expected_value, gbm_shap_values, X_test)

shap.force_plot(gbm_explainer.expected_value, gbm_shap_values, X_test)


# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(gbm_explainer.expected_value[0], gbm_shap_values[0][0,:], X_test.iloc[0,:], link="logit")


##############
#    XGB     #
##############

from xgboost import XGBClassifier
n_estimators = 500
xgb = ensemble.XGBClassifier(
            n_estimators=n_estimators,
            random_state=0)
xgb.fit(X_train, Y_train)

xgb_explainer = shap.KernelExplainer(rf.predict, X_test)
xgb_shap_values = xgb_explainer.shap_values(X_test)

shap.dependence_plot("alcohol", xgb_shap_values, X_test)

shap.force_plot(gbm_explainer.expected_value, gbm_shap_values, X_test)


##############
#    KNN     #
##############

from sklearn import neighbors
n_neighbors = 15
knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knn.fit(X_train, Y_train)

knn_explainer = shap.KernelExplainer(knn.predict, X_test)
knn_shap_values = knn_explainer.shap_values(X_test)

shap.dependence_plot("alcohol", knn_shap_values, X_test)

# plot the SHAP values for the 10th observation
shap.force_plot(knn_explainer.expected_value,knn_shap_values[10,:], X_test.iloc[10,:])

shap.force_plot(knn_explainer.expected_value, knn_shap_values, X_test)

shap.summary_plot(knn_shap_values, X_test)

##############
#    SVM     #
##############

from sklearn import svm
svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
svm.fit(X_train, Y_train)

svm_explainer = shap.KernelExplainer(svm.predict, X_test)
svm_shap_values = svm_explainer.shap_values(X_test)

shap.dependence_plot("alcohol", svm_shap_values, X_test)

# plot the SHAP values for the 10th observation
shap.force_plot(svm_explainer.expected_value,svm_shap_values[10,:], X_test.iloc[10,:])

shap.force_plot(svm_explainer.expected_value, svm_shap_values, X_test)

shap.summary_plot(svm_shap_values, X_test)


##############
#    H2O     #
##############

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
h2o.init()

X_train, X_test = train_test_split(df, test_size = 0.1)

X_train_hex = h2o.H2OFrame(X_train)
X_test_hex = h2o.H2OFrame(X_test)

X_names =  ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

# Define model
h2o_rf = H2ORandomForestEstimator(ntrees=200, max_depth=20, nfolds=10)

# Train model
h2o_rf.train(x=X_names, y='quality', training_frame=X_train_hex)

X_test = X_test_hex.drop('quality').as_data_frame()

class H2OProbWrapper:
    def __init__(self, h2o_model, feature_names):
        self.h2o_model = h2o_model
        self.feature_names = feature_names

    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        self.dataframe= pd.DataFrame(X, columns=self.feature_names)
        self.predictions = self.h2o_model.predict(h2o.H2OFrame(self.dataframe)).as_data_frame().values
        return self.predictions.astype('float64')[:,-1] #probability of True class

h2o_wrapper = H2OProbWrapper(h2o_rf,X_names)

h2o_rf_explainer = shap.KernelExplainer(h2o_wrapper.predict_binary_prob, X_test)

h2o_rf_explainer = shap.KernelExplainer(h2o_wrapper.predict_binary_prob, X_test)
h2o_rf_shap_values = h2o_rf_explainer.shap_values(X_test)

shap.summary_plot(h2o_rf_shap_values, X_test)

shap.dependence_plot("alcohol", h2o_rf_shap_values, X_test)

# plot the SHAP values for the 10th observation
shap.force_plot(h2o_rf_explainer.expected_value,h2o_rf_shap_values[10,:], X_test.iloc[10,:]) #, link="logit")

shap.force_plot(h2o_rf_explainer.expected_value, h2o_rf_shap_values, X_test)