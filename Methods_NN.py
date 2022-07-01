#Good to go

# Import external packages.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import shap

# Import local packages.
from DimensionalityBinning import DimensionalBinChoice
from SupportFunctions import PrintScores, Scoring, train_test_split, Bin_and_Standardize

# Create a neural network from a dictionary of inputs.
def KerasModel(Dict):
    # Create a sequential model.
    model = Sequential()
    # Add first hidden layer based on dictionary values.
    model.add(Dense(Dict['L1Neurons'], input_dim=Dict['input_dim'], activation=Dict['activation']))
    # Add second hidden layer based on dictionary values.
    model.add(Dense(Dict['L2Neurons'], input_dim=Dict['L1Neurons'], activation=Dict['activation']))
    # Add third hidden layer based on dictionary values.
    # model.add(Dense(Dict['L3Neurons'], input_dim=Dict['L2Neurons'], activation=Dict['activation']))
    # Add output layer.
    model.add(Dense(Dict['output'], activation=Dict['activation']))
    # Compile the model.
    model.compile(loss=Dict['loss'], optimizer=Dict['optimizer'], metrics=[Dict['metrics']])

    return model

def Model_NN(Questions, Bin_Length):
    from Methods_NN import NNParameters, KerasModel
    ParameterDictionary = NNParameters(Questions, Bin_Length)
    # y_train_level_oneHot = tf.one_hot(y_train_level, Bin_Length)
    model = KerasModel(ParameterDictionary)

    return model

# Create a neural network, train and test it using a subset of questions.
def Subset_Analysis_NN(FullQ_Train_Test_Split, ParameterDictionary_Subset, TheBinsizeList, Questionnaire_Subset):
    # FullQ_Train_Test_Split is the full set of questions which the
    X_Subset_train, X_Subset_test, y_Subset_train, y_Subset_test = FullQ_Train_Test_Split

    # SoS = Sum of Scores. Sum the scores for the questions in the training set.
    y_train_SoS = X_Subset_train.loc[:, Questionnaire_Subset].sum(axis=1)

    # Bin the sum of scores of the training set determined by the bin size in "TheBinSizeList".
    Subset_Binned_Level_y_train = DimensionalBinChoice(y_train_SoS, TheBinsizeList)

    # Create an array of the feature (question) values of questions defined in "Questionnaire_Subset".
    Train_Columns_Subset = np.asarray(X_Subset_train.loc[:, Questionnaire_Subset])
    # Create an array of target values of the training data comprised of the binned values from the subsets of questions
    # Sum of Scores (SoS).
    TrainTarget_Columns_Subset = np.asarray(Subset_Binned_Level_y_train)

    # start the binning from 0 for each of the subset binning.
    TrainTarget_Columns_Subset = TrainTarget_Columns_Subset - 1

    # Create the neural network model from the dictionary of (hyper)parameters.
    model_Subset = KerasModel(ParameterDictionary_Subset)
    # fit the model
    model_Subset.fit(Train_Columns_Subset, TrainTarget_Columns_Subset, epochs=30, batch_size=30, verbose=0)

    # Create an array of the testing feature data.
    Test_Columns_Subset = np.asarray(X_Subset_test.loc[:, Questionnaire_Subset])

    # Make a prediction from the test data.
    predictions_Subset = model_Subset.predict(Test_Columns_Subset)
    # Choose bin based on highest percentage probability.
    max_indices_Subset = np.argmax(predictions_Subset, axis=1)

    # Return the predictions.
    return max_indices_Subset

# Create and return a hyperparameter dictionary from the best found neural network (NN) hyperparameters via GridSearch.
def NNParameters(InputDim, OutputDim):
    LossFunction, Layer1_Neurons, Layer2_Neurons, ActivationFunction, Optimizer, FittingMetric = \
        'sparse_categorical_crossentropy', 50, 60, 'sigmoid', 'RMSProp', 'accuracy';

    # Create the dictionary of parameters.
    ParameterDictionary = {'input_dim': InputDim, 'activation': ActivationFunction,
                           'L1Neurons': Layer1_Neurons, 'L2Neurons': Layer2_Neurons,
                           'output': (OutputDim), 'loss': LossFunction, 'optimizer': Optimizer,
                           'metrics': FittingMetric}

    return ParameterDictionary

def Predict_NN(model, X, Y, TheBinsizeList):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)

    model.fit(X_train, Y_train, epochs=30, batch_size=30, verbose=0)
    predictions = model.predict(X_test)
    predictions_rounded = np.argmax(predictions, axis=1)
    Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions_rounded)
    ModelName = "NN"
    PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)

    return X_train, X_test, Y_train, Y_test

def SHAPPlots_NN(model, X_train, X_test, columns):

    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns[:-1]
    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = columns[:-1]

    explainer_NN = shap.DeepExplainer(model, X_train)
    shap_values_NN = explainer_NN.shap_values(X_test)

    # List = ["Summary_Single", "Summary_All", "Force_Datapoints_All", "Force_Bar", "Waterfall", "Decision"]
    List = ["Decision"]

    for Choice in List:
        if(Choice == "Summary_Single"):
            # As there are multiple shap value configurations, one for each bin, to get the scatter plot I have to choose a
            # particular configuration configuration for bin 0, so: shap_values[0] or bin 1 shap_values[1].
            # WORKS!
            shap.summary_plot(shap_values_NN[0], X_test_df)

        elif(Choice=="Summary_All"):
            # If the shap value configuration is not selected e.g. shap_values instead of shap_values[0], then all the bin
            # classes are displayed as a part of a bar graph. Otherwise, for a single shap bin configuration, if a bar graph
            # is desired, then choose plot_type="bar".
            # WORKS!
            shap.summary_plot(shap_values_NN[0], X_test_df, plot_type="bar", feature_names = X_test_df.columns)  # Gives a bar graph of multiple classes

        elif (Choice == "Force_Datapoints_All"):
            # Make the model that of a NN
            # # WORKS!
            plot = shap.force_plot(explainer_NN.expected_value[0].numpy(), shap_values_NN[0], X_test_df)
            shap.save_html("NN_All_Datapoints.htm", plot)

        elif (Choice == "Force_Bar"):
            # WORKS!
            plot = shap.force_plot(explainer_NN.expected_value[0].numpy(), shap_values_NN[0][10, :], X_test_df.iloc[10, :])
            shap.save_html("NN_Individual_Datapoint.htm", plot)

        elif (Choice == "Waterfall"):
            # Note, the difference in values between this and SVM KernelExplainer might be because I added a [0].
            shap.plots._waterfall.waterfall_legacy(explainer_NN.expected_value[0].numpy(), shap_values_NN[0][0],
                                                   feature_names=X_test_df.columns)

        elif (Choice == "Decision"):
            shap.decision_plot(explainer_NN.expected_value[0].numpy(), shap_values_NN[0][0], features=X_test_df.iloc[0, :],
                               feature_names=X_test_df.columns.tolist())

        else:
            print("Selection Not Found.")










    # shap_values_NN_df = pd.DataFrame(shap_values_NN[0], columns=columns[:-1])
    # shap_values_NN_df.feature_names = columns[:-1]
    # shap.plots.heatmap(shap_values_NN_df[1:100])






