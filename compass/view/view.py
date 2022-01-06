import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#############################################################################
def displayResults(sensitivitiesTrain, specificitiesTrain, AUCsTrain, AccuTrain, finalSensitivity, finalSpecificity,
                   finalAUC, finalAccu, finalMCC, mccTrain, trainRatio):
    #############################################################################

    """
    Diplays results on the different metrics of Accuracy, AUC, number of features, etc.

    :param sensitivitiesTrain
    :param specificitiesTrain
    :param AUCsTrain
    :param AccuTrain
    :param finalSensitivity
    :param finalSpecificity
    :param finalAUC
    :param finalAccu
    :param finalMCC
    :param mccTrain
    :param trainRatio
    :return:none

    """

    print("\n\n")
    print("==============================================")

    if trainRatio < 1:
        print("================ train set ===================")
    else:
        print("================= CV set =====================")
    print("Sensitivity          :", str(np.round(np.mean(sensitivitiesTrain) * 100, 3)).center(7), "%    +/- ",
          np.round(np.std(sensitivitiesTrain) * 100, 2))
    print("Specificity          :", str(np.round(np.mean(specificitiesTrain) * 100, 3)).center(7), "%    +/- ",
          np.round(np.std(specificitiesTrain) * 100, 3))
    print("Accuracy             :", str(np.round(np.mean(AccuTrain), 3)).center(7), " %", "   +/- ",
          np.round(np.std(AccuTrain), 3))
    print("MCC                  :", str(np.round(np.mean(mccTrain), 3)).center(7), "", "    +/- ",
          np.round(np.std(mccTrain), 3))
    print("AUC                  :", str(np.round(np.mean(AUCsTrain), 3)).center(7), "     +/- ",
          np.round(np.std(AUCsTrain), 3))

    if trainRatio < 1:
        ### Note that test set final is useful for trainRatio<1
        print("============= final test set =================")
        print("Sensitivity          :", np.round(finalSensitivity, 3) * 100, "%")
        print("Specificity          :", np.round(finalSpecificity, 3) * 100, "%")
        print("Accuracy             :", np.round(finalAccu, 3), " %")
        print("MCC                  :", np.round(finalMCC, 3), "")
        print("AUC                  :", np.round(finalAUC, 3))


def FinalDisplay(bestAucT, bestSensT, bestSensStdT, bestSpeT, bestSpeStdT, bestAccT, bestAccStdT,
                 bestMccT, bestMccStdT, bestSens, bestSpe, bestAcc, bestMcc, bestAuc, angleType, nbVariables, fileName,
                 numgen, timing, angleTheta, trainRatio):

    """
    Diplays final results on the different metrics of Accuracy, AUC, number of features, etc.
    :param bestAucT
    :param bestSensT
    :param bestSensStdT
    :param bestSpeT
    :param bestSpeStdT
    :param bestAccT
    :param bestAccStdT
    :param bestMccT
    :param bestMccStdT
    :param bestSens
    :param bestSpe
    :param bestAcc
    :param bestMcc
    :param bestAuc
    :param angleType
    :param nbVariables
    :param fileName
    :param numgen
    :param timing
    :param angleTheta
    :param trainRatio
    :return:none

    """

    print("==============================================")

    if (trainRatio < 1):
        print("================ train set ===================")
    else:
        print("================= CV set =====================")
    print("Sensitivity          :", str(bestSensT).center(7), "%    +/- ", bestSensStdT)
    print("Specificity          :", str(bestSpeT).center(7), "%    +/- ", bestSpeStdT)
    print("Accuracy             :", str(bestAccT).center(7), "%", "   +/- ", bestAccStdT)
    print("MCC                  :", str(bestMccT).center(7), "", "    +/- ", bestMccStdT)
    print("AUC                  :", str(np.round(np.mean(bestAucT), 3)).center(7), "     +/- ",
          np.round(np.std(bestAucT), 3))

    if (trainRatio < 1):
        ### Note that test set final is useful for trainRatio<1 only
        print("============= final test set =================")
        print("Sensitivity            :", bestSens, "%")
        print("Specificity            :", bestSpe, "%")
        print("Accuracy               :", bestAcc, "%")
        print("MCC                    :", bestMcc, "")
        print("AUC                    :", bestAuc)

    print("==============================================")
    print("ANGLE TYPE             :", angleType)
    print("THETA                  :", angleTheta)
    print("Number of features     :", nbVariables)
    print("Generation             :", numgen)
    print("Excecution time        : ", timing)
    print("File Name              :", fileName)


def DisplayParams(CROSS_VALIDATIONS_NUMBER, NORMALIZATION, DATASET, ML_METHOD,
                  INDIVIDUALS_NUMBER, INDIVIDUAL_INJECTION, DIMENSION_MODE, ACCU, NB_RUN, PATHRES,
                  trainRatio, datasetName, MAXGEN, NBFeatRatio, RES_FILE, EXPL_TYPE, RANGE,SEED):
    """
    Diplays experiment parameters

    :param CROSS_VALIDATIONS_NUMBER
    :param NORMALIZATION
    :param DATASET
    :param ML_METHOD
    :param INDIVIDUALS_NUMBER
    :param INDIVIDUAL_INJECTION
    :param DIMENSION_MODE
    :param ACCU
    :param NB_RUN
    :param PATHRES
    :param trainRatio
    :param datasetName
    :param MAXGEN
    :param NBFeatRatio
    :param RES_FILE
    :param EXPL_TYPE
    :param RANGE
    :return: none

    """


    print("\n")
    print("DATASET : ", datasetName)
    print("Folder in ", PATHRES, "  : ", DATASET)
    print("CROSS VALIDATIONS NUMBER : ", CROSS_VALIDATIONS_NUMBER)
    print("NORMALIZATION            : ", NORMALIZATION)
    print("METHOD                   : ", ML_METHOD)
    print("INDIVIDUAL NUMBER        : ", INDIVIDUALS_NUMBER)
    print("INDIVIDUAL INJECTION     : ", INDIVIDUAL_INJECTION)
    print("DIMENSION MODE           : ", DIMENSION_MODE)
    print("ACCU                     : ", ACCU)
    print("NUMBER OF RUN            : ", NB_RUN)
    print("RESULTS PATH             : ", PATHRES)
    print("TRAINING RATIO           : ", trainRatio * 100, "%")
    print("MAX GENERATION           : ", MAXGEN)
    print("FEATURES RATIO FACTOR    : ", NBFeatRatio)
    print("RESULTS FILE NAME        : ", RES_FILE)
    print("EXPLORATION MODE         : ", EXPL_TYPE)
    print("RANGE MODE               : ", RANGE)
    if SEED>-1:
        print("SEED                     : ", SEED)
    else:
        print("SEED                     : ", "RANDOM")

def DisplayObjective(DIMENSION_MODE, THETA, targetedAUC):
    """
    Displays experiments objective

    :param DIMENSION_MODE
    :param THETA
    :param targetedAUC
    :return: none
    """


    print("\n\nOBJECTIVE(S) MODE IS " + DIMENSION_MODE)

    if (DIMENSION_MODE == "2D"):
        print("Theta: " + str(THETA))
    if (DIMENSION_MODE == "3D"):
        print("Targeted AUC: " + str(targetedAUC))
        print("Angle Var-Acc: " + str(THETA) + '°\n')


def DisplayAngleType(degrees):
    """
    Displays angle type
    :param degrees
    :return:none
    """
    print("Angle Type: " + str(degrees))


def DiplaySeeds(seeds):
    """
    Displays used seed(s)
    :param seeds
    :return:none
    """
    print("\nUSED SEEDS:")
    print(seeds)


def DisplayRun(nbRun):
    """
    Displays the run number
    :param nbRun
    :return:none
    """
    print("\n")
    print("You are processing in run number : ", nbRun)


def DisplaySelectedFeatures(datasetFileName, bestIndivFileName, TOTAL_NBR_FEATURES):
    """
    Displays selected features name
    :param datasetFileName
    :param bestIndivFileName
    :param TOTAL_NBR_FEATURES
    :return:none

    """
    best_selection = np.load(bestIndivFileName)

    df = pd.read_excel("./compass/data/" + datasetFileName, engine='openpyxl')
    features_name = df.columns

    df = pd.read_excel("./compass/filesManagement/ColumnsNames.xlsx", engine='openpyxl')
    noms_variables_excel = df.columns

    print('\n -------- SELECTED FEATURES --------')
    print('column_index ) selected_features_name :')
    print('--------------------------')
    for i in range(TOTAL_NBR_FEATURES - 1):
        if best_selection[i] == 1:
            print(str(noms_variables_excel[i]) + ' ) ' + str(features_name[i]))


def DisplayStats(accTabGraph, nbVarTabGraph, aucTabGraph, degTabGraph, stopTime, startTime):
    """
    Display statitics Min, Max, Average. Displats tables of angles, number of features, accuracies, AUCs of all
    experiments. Displays the total exploration time.

    :param accTabGraph
    :param nbVarTabGraph
    :param aucTabGraph
    :param degTabGraph
    :param stopTime
    :param startTime
    :return:none
    """
    print("----------------\n")
    print("BEST    ---> ", "Accuracy: ", max(accTabGraph), " / Number of features: (", min(nbVarTabGraph), ") / AUC: ",
          round(max(aucTabGraph), 2))
    print("AVERAGE ---> ", "Accuracy: ", round(np.mean(accTabGraph),3), " / Number of features: (", np.mean(nbVarTabGraph),
          ") / AUC: ", round(np.mean(aucTabGraph), 2))
    print("WORST   ---> ", "Accuracy: ", min(accTabGraph), " / Number of features: (", max(nbVarTabGraph), ") / AUC: ",
          round(min(aucTabGraph), 2))
    print("\n")
    print("Angles: ", degTabGraph)
    print("Numbers of features: ", nbVarTabGraph)
    print("Accuracies: ", accTabGraph)
    print("AUCs: ", aucTabGraph)
    print("Total exploration time:" + str(stopTime - startTime))


def TraceGraph(vx, vy, name, colorG, DATASET):
    """
    Traces scatter plots and saves in figures
    :param vx
    :param vy
    :param name
    :param colorG
    :param DATASET
    :return:none
    """
    fig = plt.figure()

    axes = plt.gca()
    axes.set_xlim(0, max(vx) + 1)
    axes.set_ylim(0, max(vy) + max(vy) / 10)

    """
    Labels and title
    """
    plt.xlabel("Number of features", size=16)
    if (name == "Accuracy"):
        plt.ylabel(name + " (%)", size=16)
    elif (name == "Angle"):
        plt.ylabel(name + " (°)", size=16)
    else:
        plt.ylabel(name, size=16)

    """
    Trace all points
    """
    plt.plot(vx, vy, '.', color=colorG)
    # the third arg '.' allows to precise that we want a scatter graph

    fig.savefig('./compass/results/' + DATASET + '/' + name + '.svg')
    fig.savefig('./compass/results/' + DATASET + '/' + name + '.png')


def generateTable(vangle, vx, vy, vz, DATASET):
    """
    Generate a table of data to visualize in png or svg
    :param vangle
    :param vx
    :param vy
    :param vz
    :param DATASET
    :return:none
    """
    data = []
    lab = ['Angle', 'nbFeat', 'Accu', 'AUC']
    for i in range(len(vangle)):
        data.append([vangle[i], vx[i], vy[i], vz[i]])

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame(data, columns=lab)

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()

    fig.savefig('./compass/results/' + DATASET + '/' + 'table.png')
    fig.savefig('./compass/results/' + DATASET + '/' + 'table.svg')
