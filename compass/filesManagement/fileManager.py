import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Pour normaliser les donnÃ©es


#############################################################################
def readCorpus(nomFichier, trainRatio, NORM):
    #############################################################################
    #############################################################################
    """ will read from the file: "fileName.xlsx"


    the data to create:

        - the explanatory variables of the learning set: x_train
        - the variable to be predicted from the training set: y_train

        - the explanatory variables of the final test set: x_test_final
        - the variable to predict from the final test set: y_test_final

        - call it like this:
            x_train, y_train, x_test_final, y_test_final = readCorpus()

        :param nomFichier
        :param trainRatio
        :param NORM
        :return: x_train, y_train, x_test_final, y_test_final, number_columns_base
        :rtype: list or integer
        
    """

    print("\nExperiment on " + nomFichier)

    df = pd.read_excel("./compass/data/" + nomFichier, engine='openpyxl')

    total_set_data = df.values

    number_lines_base = total_set_data.shape[0]
    number_columns_base = total_set_data.shape[1]

    ###########################################################################################♣

    cutTrainTest = round(number_lines_base * trainRatio)  # a quel endroit on coupe entre apprentissage/test
    x_train = total_set_data[0:cutTrainTest, 0:number_columns_base - 1]
    y_train = total_set_data[0:cutTrainTest, number_columns_base - 1]

    if (trainRatio == 1):
        x_test_final = x_train.copy()  # total_set_data[cutTrainTest:,0:number_columns_base-1]
        y_test_final = y_train.copy()  # total_set_data [cutTrainTest:,number_columns_base-1]
    else:
        x_test_final = total_set_data[cutTrainTest:, 0:number_columns_base - 1]
        y_test_final = total_set_data[cutTrainTest:, number_columns_base - 1]

    # data normalization
    if NORM:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test_final = scaler.transform(x_test_final)

    return x_train, y_train, x_test_final, y_test_final, number_columns_base


def mixBase(x_train, y_train, x_test_final, y_test_final, seed):
    """
    Mix the base to ensure prevalence
    :param x_train
    :param y_train
    :param x_test_final
    :param y_test_final
    :param seed
    :return: x_train, y_train, x_test_final, y_test_final
    :rtype:list
    """

    np.random.seed(seed)

    for i in range(100000):
        # switch inside the train set (useless in the final test !!)
        index1 = np.random.randint(0, round(x_train.shape[0]) - 1)
        index2 = np.random.randint(0, round(x_train.shape[0]) - 1)

        for j in range(x_train.shape[1]):
            temp = x_train[index1, j]
            x_train[index1, j] = x_train[index2, j]
            x_train[index2, j] = temp

        temp = y_train[index1]
        y_train[index1] = y_train[index2]
        y_train[index2] = temp

        # switch between the train set and the final test set
        index1 = np.random.randint(0, round(x_train.shape[0]) - 1)
        index2 = np.random.randint(0, round(x_test_final.shape[0]) - 1)

        if y_train[index1] == y_test_final[index2]:
            for j in range(x_train.shape[1]):
                temp = x_train[index1, j]
                x_train[index1, j] = x_test_final[index2, j]
                x_test_final[index2, j] = temp

    return x_train, y_train, x_test_final, y_test_final


def saveFiles(PATHRES, DATASET, ML_METHOD, fileName, population, fitness, sensitivitiesTrain, specificitiesTrain,
              AUCsTrain, AccuTrain, finalSensitivity, finalSpecificity, finalAUC, finalAccu):

    """
    Saves files .npy, population, best individual and last individual for further possible use
    :param PATHRES
    :param DATASET
    :param ML_METHOD
    :param fileName
    :param population
    :param fitness
    :param sensitivitiesTrain
    :param specificitiesTrain
    :param AUCsTrain
    :param AccuTrain
    :param finalSensitivity
    :param finalSpecificity
    :param finalAUC
    :param finalAccu
    :return: none

    """

    np.save(fileName + '---Population', population)
    np.save(fileName + '---BestIndividual', population[np.argmax(fitness)])
    savePerfs(PATHRES, DATASET, ML_METHOD, sensitivitiesTrain, specificitiesTrain, AUCsTrain, AccuTrain,
              finalSensitivity, finalSpecificity, finalAUC, finalAccu, population[np.argmax(fitness)])

    np.save('./compass/' + PATHRES + DATASET + '/LastIndividual', population[np.argmax(fitness)])


def fileNameBuilder(population, PATHRES, DATASET, ML_METHOD, fitness, numGeneration, sensitivitiesTrain,
                    specificitiesTrain, AUCsTrain, AccuTrain, finalSensitivity, finalSpecificity, finalAUC,
                    finalAccu, run, angle):
    """
    Build results file name (.npy)
    :param population
    :param PATHRES
    :param DATASET
    :param ML_METHOD
    :param fitness
    :param numGeneration
    :param sensitivitiesTrain
    :param specificitiesTrain
    :param AUCsTrain
    :param AccuTrain
    :param finalSensitivity
    :param finalSpecificity
    :param finalAUC
    :param finalAccu
    :param run
    :param angle
    :return:fileName
    :rtype:string
    """
    fileName = './compass' + PATHRES + DATASET + '/' + ML_METHOD + "-" + "-"
    fileName = fileName + ' ---RUN ' + str(run) + ' ---ANGLE ' + str(math.degrees(angle)) + ' ---GEN ' + str(
        numGeneration)

    ### If wanna use other informations like population, fitness, sensitivitiesTrain, etc. they are available as inputs

    return fileName


#############################################################################
def savePerfs(PATHRES, DATASET, ML_METHOD, sensitivitiesTrain, specificitiesTrain, AUCsTrain, AccuTrain,
              finalSensitivity, finalSpecificity, finalAUC, finalAccu, selectedFeatures):
    #############################################################################

    """
    Saves files .npy, obtaining better performance than a previous generation
    :param PATHRES
    :param DATASET
    :param ML_METHOD
    :param sensitivitiesTrain
    :param specificitiesTrain
    :param AUCsTrain
    :param AccuTrain
    :param finalSensitivity
    :param finalSpecificity
    :param finalAUC
    :param finalAccu
    :param selectedFeatures
    :return: none

    """

    fileName = './compass/' + PATHRES + DATASET + '/---' + ML_METHOD + "-" + "-"
    temp = str(round(np.mean(AUCsTrain) * 1000) / 1000)
    while (len(temp) < 5):
        temp = temp + "0"

    fileName = fileName + '--AUC ' + temp

    temp = str(round(np.std(AUCsTrain) * 1000) / 1000)
    while (len(temp) < 5):
        temp = temp + "0"
    fileName = fileName + ' +- ' + temp

    fileName = fileName + '  ---Accu ' + str(round(np.mean(AccuTrain), 2))
    fileName = fileName + '  +- ' + str(round(np.std(AccuTrain), 2))

    fileName = fileName + '  ---Se ' + str(round(np.mean(sensitivitiesTrain) * 100))
    fileName = fileName + '  +- ' + str(round(np.std(sensitivitiesTrain) * 100, 2))

    fileName = fileName + '  ---Sp ' + str(round(np.mean(specificitiesTrain) * 100, 2))
    fileName = fileName + '  +- ' + str(round(np.std(specificitiesTrain) * 100, 2))

    ## If wanna add TEST informations in the name of the file
    # fileName = fileName + '  --TEST--   '
    # fileName = fileName + '  AUC ' + str(finalAUC)
    # fileName = fileName + '  ACC ' + str(finalAccu)
    # fileName = fileName + '  Se ' + str(round(finalSensitivity * 100))
    # fileName = fileName + '  Sp ' + str(round(finalSpecificity * 100))
    ##
    fileName = fileName + '---NBR ' + str(sum(selectedFeatures))

    np.save(fileName, selectedFeatures)


def saveFinalPerf(PATHRES, DATASET, acc, auc, nbFeat):

    """
    Saves final performances in a .csv file

    :param PATHRES
    :param DATASET
    :param acc
    :param auc
    :param nbFeat
    :return: none


    """

    fileName = './compass/' + PATHRES + DATASET + '/' + DATASET + '.csv'

    f = open(fileName, 'w')

    for i in range(len(acc)):

        if (i != len(acc) - 1):

            f.write(str(acc[i]) + ',')

        else:

            f.write(str(acc[i]) + '\n')

    for i in range(len(auc)):

        if (i != len(auc) - 1):

            f.write(str(auc[i]) + ',')

        else:

            f.write(str(auc[i]) + '\n')

    for i in range(len(nbFeat)):

        if (i != len(nbFeat) - 1):

            f.write(str(nbFeat[i]) + ',')

        else:

            f.write(str(nbFeat[i]) + '\n')

    f.close()
