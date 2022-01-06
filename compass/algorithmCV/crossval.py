# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:29:45 2021

@author: n.gutowski
"""
import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from compass.methods.mlMethods import myModel


############################################################################
def computeSensitivitiesSpecificitiesAUCsAccus(cross_validations_number, x_train, y_train, mlMethod):
    ############################################################################
    """
    For a particular machine learning method, calculation of sensitivities, specificities, AUC and accuracies through
    cross validations on all the dataset.

    :param cross_validations_number
    :param x_train
    :param y_train
    :param mlMethod

    :return: sensitivitiesTrain, specificitiesTrain, aucsTrain, AccuTrain, finalSensitivity, finalSpecificity, aucFinale, finalAccu, mccTrain, mccFinale
    :rtype: list of float
    """
    model = myModel(mlMethod)
    sensitivitiesTrain = []
    specificitiesTrain = []
    aucsTrain = []
    AccuTrain = []
    mccTrain = []

    cross_validation = StratifiedKFold(n_splits=cross_validations_number)

    for train, test in cross_validation.split(x_train, y_train):

        model.fit(x_train[train], y_train[train])

        if mlMethod == "BR" or mlMethod == "RID" or mlMethod == "RNPER" or mlMethod == "ELAS" or mlMethod == "LASSO" or mlMethod == "RN" or mlMethod == "ELM":
            y_train_predit_test = model.predict(x_train[test])
            confusionMatrix = metrics.confusion_matrix(y_train[test], np.round(y_train_predit_test))
            aucsTrain.append(metrics.roc_auc_score(y_train[test], y_train_predit_test))
            mccTrain.append(matthews_corrcoef(y_train[test], np.round(y_train_predit_test)))

        else:
            probas_y_train_test = model.predict_proba(x_train[test])
            confusionMatrix = metrics.confusion_matrix(y_train[test], np.round(probas_y_train_test[:, 1]))
            aucsTrain.append(metrics.roc_auc_score(y_train[test], probas_y_train_test[:, 1]))
            mccTrain.append(matthews_corrcoef(y_train[test], np.round(probas_y_train_test[:, 1])))

        VN = confusionMatrix[0, 0]  # ok, vérifié
        FP = confusionMatrix[0, 1]
        FN = confusionMatrix[1, 0]
        VP = confusionMatrix[1, 1]

        sensitivitiesTrain.append(VP / (VP + FN))
        specificitiesTrain.append(VN / (VN + FP))
        AccuTrain.append((VP + VN) / (VP + VN + FP + FN) * 100)

    return sensitivitiesTrain, specificitiesTrain, aucsTrain, AccuTrain, mccTrain


#############################################################################
def computeSensitivitiesSpecificitiesAUCsAccusAndFinals(cross_validations_Number, x_train, y_train, x_test_final,
                                                        y_test_final, mlMethod):
    #############################################################################
    """
    For a particular machine learning method, calculation of sensitivities, specificities, AUC and accuracies through
    cross validations on for example 75% of the dataset (depends on the ratio parameter).
    Calculation of the sensitivity, the specificity, the AUC and the accuracy in the rest of the dataset
    (for example in the 25% remaining of the dataset).

    :param cross_validations_Number
    :param x_train
    :param y_train
    :param x_test_final
    :param y_test_final
    :param mlMethod

    :return: sensitivitiesTrain, specificitiesTrain, aucsTrain, AccuTrain, finalSensitivity, finalSpecificity, aucFinale, finalAccu, mccTrain, mccFinale
    :rtype: float or list of float
    """

    model = myModel(mlMethod)
    sensitivitiesTrain = []
    specificitiesTrain = []
    aucsTrain = []
    AccuTrain = []
    mccTrain = []

    cross_validation = StratifiedKFold(n_splits=cross_validations_Number)

    for train, test in cross_validation.split(x_train, y_train):

        model.fit(x_train[train], y_train[train])

        if mlMethod == "BR" or mlMethod == "RID" or mlMethod == "RNPER" or mlMethod == "ELAS" or mlMethod == "LASSO" or mlMethod == "RN" or mlMethod == "ELM":
            y_train_predit_test = model.predict(x_train[test])
            confusionMatrix = metrics.confusion_matrix(y_train[test], np.round(y_train_predit_test))
            aucsTrain.append(metrics.roc_auc_score(y_train[test], y_train_predit_test))
            mccTrain.append(matthews_corrcoef(y_train[test], np.round(y_train_predit_test)))

        else:
            probas_y_train_test = model.predict_proba(x_train[test])
            confusionMatrix = metrics.confusion_matrix(y_train[test], np.round(probas_y_train_test[:, 1]))
            aucsTrain.append(metrics.roc_auc_score(y_train[test], probas_y_train_test[:, 1]))
            mccTrain.append(matthews_corrcoef(y_train[test], np.round(probas_y_train_test[:, 1])))

        VN = confusionMatrix[0, 0]
        FP = confusionMatrix[0, 1]
        FN = confusionMatrix[1, 0]
        VP = confusionMatrix[1, 1]

        sensitivitiesTrain.append(VP / (VP + FN))
        specificitiesTrain.append(VN / (VN + FP))
        AccuTrain.append((VP + VN) / (VP + VN + FP + FN) * 100)

        model.fit(x_train, y_train)

        if mlMethod == "BR" or mlMethod == "RID" or mlMethod == "RNPER" or mlMethod == "ELAS" or mlMethod == "LASSO" or mlMethod == "RN" or mlMethod == "ELM":
            y_predit_test_final = model.predict(x_test_final)
            confusionMatrix = metrics.confusion_matrix(y_test_final, np.round(y_predit_test_final))
            aucFinale = metrics.roc_auc_score(y_test_final, y_predit_test_final)
            mccFinale = matthews_corrcoef(y_test_final, np.round(y_predit_test_final))

        else:
            probas_y_test_final = model.predict_proba(x_test_final)
            confusionMatrix = metrics.confusion_matrix(y_test_final, np.round(probas_y_test_final[:, 1]))
            aucFinale = metrics.roc_auc_score(y_test_final, probas_y_test_final[:, 1])
            mccFinale = matthews_corrcoef(y_test_final, np.round(probas_y_test_final[:, 1]))

        VN = confusionMatrix[0, 0]  # ok, vérifié
        FP = confusionMatrix[0, 1]
        FN = confusionMatrix[1, 0]
        VP = confusionMatrix[1, 1]

        finalSensitivity = (VP / (VP + FN))
        finalSpecificity = (VN / (VN + FP))
        finalAccu = ((VP + VN) / (VP + VN + FP + FN)) * 100

    return sensitivitiesTrain, specificitiesTrain, aucsTrain, AccuTrain, finalSensitivity, finalSpecificity, aucFinale, finalAccu, mccTrain, mccFinale
