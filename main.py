
from pandas import DataFrame
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Import local methods
from SyntheticMethods import syntheticdata
from SupportFunctions import BinConfigurations, Kfold

from Methods_NN import Predict_NN, SHAPPlots_NN, Model_NN
from Methods_SVM import Predict_SVM, SHAPPlots_SVM, Model_SVM
from Methods_RFR import Predict_RFR, SHAPPlots_RFR, Model_RFR
from Methods_EGB import Predict_EGB, SHAPPlots_EGB, Model_EGB
from Methods_KNN import Predict_KNN, SHAPPlots_KNN, Model_KNN
from Methods_H20RF import Predict_H20RF, SHAPPlots_H20RF, Model_H20RF


def ChooseMethod(Choice):
    if(Choice == "NN"):
        # Make the model that of a NN
        model = Model_NN(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_NN(model, X, Y, Bin)
        SHAPPlots_NN(model, X_train, X_test, df_Columns)
    elif(Choice == "SVM"):
        model = Model_SVM(Questions,Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_SVM(model, X, Y, Bin)
        SHAPPlots_SVM(model, X_train, X_test, df_Columns)
    elif (Choice == "RFR"):
        model = Model_RFR(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_RFR(model, X, Y, Bin)
        SHAPPlots_RFR(model, X_train, X_test, df_Columns)
    elif (Choice == "EGB"):
        model = Model_EGB(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_EGB(model, X, Y, Bin)
        SHAPPlots_EGB(model, X_train, X_test, df_Columns)
    elif (Choice == "KNN"):
        model = Model_KNN(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_KNN(model, X, Y, Bin)
        SHAPPlots_KNN(model, X_train, X_test, df_Columns)
    elif (Choice == "H20_RF"):
        model = Model_H20RF(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_H20RF(model, X, Y, Bin, df_Columns)
        SHAPPlots_H20RF(model, X_train, X_test, df_Columns)
    else:
        print("Selection Not Found.")


# prepare cross validation
kfold = KFold(n_splits=10, shuffle=False, random_state=None)
# enumerate splits

Questions = 3
Subjects = 2000

QuestionList = []
for i in range(Questions):
    QuestionList.append("Q_" + str(i+1))


syntheticDatachoice = 1  # 1 = Dirichlet number generation
SyntheticData = syntheticdata(syntheticDatachoice, Subjects, Questions)
df = DataFrame.from_records(SyntheticData)
df.columns = QuestionList
df["Sum_of_Scores"]=df.iloc[:,:].sum(axis=1)
df_Columns = df.columns

Bin = BinConfigurations()


# Bins = len(Bin)+ 1
Bin_Length = len(Bin) + 1


X = df.iloc[:, 0:-1].values
Y = df.iloc[:, -1:].values

# ChooseMethod("NN")
# ChooseMethod("SVM")
# ChooseMethod("RFR")
# ChooseMethod("EGB")
# ChooseMethod("KNN")



# H20 throws an error.
# ChooseMethod("H20_RF")



# Uncomment if Kfold is wanted:
# Kfold(model, X, Y, Bin)



# X_train_df = pd.DataFrame(X_train, columns = QuestionList)
# x_test_df = pd.DataFrame(X_test, columns = QuestionList)


