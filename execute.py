import math
import random as rnd
import sys
import warnings
from datetime import datetime

import numpy as np

import compass.algorithmCV.GA as GA
import compass.algorithmCV.crossval as cv
from compass.filesManagement.fileManager import readCorpus, mixBase, saveFiles, fileNameBuilder, saveFinalPerf
from compass.methods.buildCompass import buildCompass
from compass.methods.init import initialize, initializeFinalMetrics, parameters
from compass.methods.utilMethods import setAngle, setMAXDEG, keepBestPerf
from compass.view.view import displayResults, FinalDisplay, DisplayParams, DisplayAngleType, \
    DisplayObjective, \
    DiplaySeeds, DisplayRun, DisplaySelectedFeatures, DisplayStats, TraceGraph, generateTable


##############################################################################
#                             MAIN PROGRAM :                                 #
##############################################################################

def main(jsonfile="./compass/params/" + sys.argv[1]):

    """
    This is the main program to run. Parameters are loaded from JSON parameters file
    'jsonfile="./compass/params/"+ sys.argv[1]' where sys.argv[1] is the file name given as argument.

    The main program is the centerpiece that makes the whole GAwC running. It also saves various results files :
    - Population (.npy) ;
    - Best Individual (.npy) ;
    - Last Individual (.npy, allowing to stop and run from the last individual if necessary with individual injection mode) ;
    - Figures ;
    - Tables in textual and graphical representation ;
    - Used seed(s) ;
    - Running time ;
    - All accuracies, AUCs and numbers of selected features (in .csv and in the .txt results file) ;
    - Statistics like average, max and min results in terms of accuracy, AUC and number of selected features.

    Experiment parameters are:
    Please see README for mor information on parameters.

    :return:None
    """




    ### Activate if needed for some methods to try
    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    ### Loads experiment parameters loading
    CROSS_VALIDATIONS_NUMBER, NORMALIZATION, DATASET, ML_METHOD, INDIVIDUALS_NUMBER, INDIVIDUAL_INJECTION, DIMENSION_MODE, ACCU, NB_RUN, PATHRES, TRAIN_RATIO, DATASET_NAME, MAXGEN, NBFEAT_RATIO, RES_FILE, EXPL_TYPE, RANGE,SEED = parameters(
        jsonfile)
    ### Displays experiment parameters
    DisplayParams(CROSS_VALIDATIONS_NUMBER, NORMALIZATION, DATASET, ML_METHOD, INDIVIDUALS_NUMBER, INDIVIDUAL_INJECTION,
                  DIMENSION_MODE, ACCU, NB_RUN, PATHRES, TRAIN_RATIO, DATASET_NAME, MAXGEN, NBFEAT_RATIO, RES_FILE,
                  EXPL_TYPE,
                  RANGE,SEED)

    ### Initializes final best metrics to keep at end after iterative comparison
    BESTPERF, bestAucT, bestAucStd, bestSensT, bestSensStdT, bestSpeT, bestSpeStdT, bestAccT, bestAccStdT, bestMccT, bestMccStdT, bestSens, bestSpe, bestAcc, bestMcc, bestAuc, angle, nbFeatures, bestFileName = initialize()

    ### Initializes final metrics array that will allow to compute final statistics (Min,Max,Average)
    accTab, aucTab, nbVarTab, seedsTab, accTabGraph, aucTabGraph, nbVarTabGraph, degTabGraph = initializeFinalMetrics()

    ### All results will be saved in ./compass/results/ with the folder name and file name given in parameter
    ### Do not forget to manualy create the folder
    if (RES_FILE != "none"):
        ### All prints will be stored into the results file
        orig_stdout = sys.stdout
        f = open("./compass/results/" + DATASET + "/" + RES_FILE, 'w')
        sys.stdout = f

    ### For loop starts from 0 to NB_RUN
    for iteration in range(NB_RUN):

        DisplayRun(iteration)
        x_train, y_train, x_test_final, y_test_final, TOTAL_NBR_FEATURES = readCorpus(DATASET_NAME, TRAIN_RATIO,
                                                                                      NORMALIZATION)

        ### Seeds are essential to run again a same experiment. It is thus possible to generate it at random (put -1
        ### in the JSON file seed parameter) or to choose a specific one e.g., seed=42, seed=845.
        if (SEED>-1):
            seed=SEED
        else:
            seed = rnd.randint(0, 1000)
        seedsTab.append(seed)

        ### Mix the base to ensure prevalence
        x_train, y_train, x_test_final, y_test_final = mixBase(x_train, y_train, x_test_final, y_test_final,
                                                               seed)

        INDIVIDUAL_DIMENSION = x_train.shape[1]

        start_BeforeLooptime = datetime.now()

        MAX_DEG = setMAXDEG(EXPL_TYPE, RANGE)


        ### Depending on how the compass exploration is set, it will process exploration degree by degree (from the
        ### beginning (degrees) to the end (MAX_DEG).
        for degrees in range(MAX_DEG):

            newGenCounter = 0

            ### Initialize intermediary best metrics (degree) to keep after iterative comparison at the end of the
            ### degree processing
            interBESTPERF, interbestAucT, interbestAucStd, interbestSensT, interbestSensStdT, interbestSpeT, interbestSpeStdT, interbestAccT, interbestAccStdT, interbestMccT, interbestMccStdT, interbestSens, interbestSpe, interbestAcc, interbestMcc, interbestAuc, interangle, internbFeatures, interfileName = initialize()

            start_looptime = datetime.now()

            ### Defines the angle as parameterized in the JSON parameters file
            angle = setAngle(degrees, EXPL_TYPE, MAX_DEG, RANGE)

            ### Set the compass as parameterized in the JSON parameters file
            COMPASS, THETA, targetedAUC = buildCompass(DIMENSION_MODE, angle)

            ### Diplays experiments informations
            DisplayObjective(DIMENSION_MODE, THETA, targetedAUC)
            DisplayAngleType(degrees)

            ### Population initialization
            print("Population initialization starts...")
            population, fitness = GA.initPopulation(INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION, x_train, y_train,
                                                    INDIVIDUAL_INJECTION, CROSS_VALIDATIONS_NUMBER, ML_METHOD,
                                                    COMPASS, ACCU, DIMENSION_MODE, NBFEAT_RATIO)

            numGeneration = 0
            start_time = datetime.now()
            bestIndividualFitness = -1

            while (True):
                print(numGeneration, " / ", end='')

                ### Tournament Selection
                parentsPopulation, parentsFitness = GA.tournamentSelection(population, fitness,
                                                                           INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION)
                ### Crossing
                childrenPopulation, childrenFitness = GA.crossingallParents(parentsPopulation, parentsFitness,
                                                                            INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION,
                                                                            x_train, y_train,
                                                                            CROSS_VALIDATIONS_NUMBER,
                                                                            ML_METHOD, COMPASS, ACCU,
                                                                            DIMENSION_MODE, NBFEAT_RATIO)

                population = np.concatenate((parentsPopulation, childrenPopulation), axis=0)
                fitness = np.concatenate((parentsFitness, childrenFitness), axis=0)

                ### Mutation
                population, fitness = GA.mutation(population, fitness, INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION,
                                                  50, np.argmax(fitness), x_train, y_train,
                                                  CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE,
                                                  NBFEAT_RATIO)

                ### Comparison on the fitness
                if (bestIndividualFitness < max(fitness)):
                    bestIndividualFitness = max(fitness)

                    ### Features selection
                    x_train_res, x_test_final_res = GA.featuresandFinalsSelection(population[np.argmax(fitness)],
                                                                                  x_train,
                                                                                  x_test_final)
                    ### Metrics computation
                    sensitivitiesTrain, specificitiesTrain, AUCsTrain, accuTrain, finalSensitivity, finalSpecificity, finalAUC, finalAccu, mccTrain, finalMCC = cv.computeSensitivitiesSpecificitiesAUCsAccusAndFinals(
                        CROSS_VALIDATIONS_NUMBER, x_train_res, y_train, x_test_final_res, y_test_final, ML_METHOD)
                    displayResults(sensitivitiesTrain, specificitiesTrain, AUCsTrain, accuTrain, finalSensitivity,
                                   finalSpecificity, finalAUC, finalAccu, finalMCC, mccTrain, TRAIN_RATIO)

                    ### fileName building (for .npy files)
                    fileName = fileNameBuilder(population, PATHRES, DATASET, ML_METHOD, fitness, numGeneration,
                                               sensitivitiesTrain, specificitiesTrain, AUCsTrain, accuTrain,
                                               finalSensitivity, finalSpecificity, finalAUC, finalAccu, iteration + 1,
                                               angle)
                    ### Saving .npy files
                    saveFiles(PATHRES, DATASET, ML_METHOD, fileName, population, fitness, sensitivitiesTrain,
                              specificitiesTrain, AUCsTrain, accuTrain, finalSensitivity, finalSpecificity, finalAUC,
                              finalAccu)

                ### Computes and keep the processing time
                diffDate = str(datetime.now() - start_time)

                ### If ACCU is true, the comparisonCriteria will be Accuracy. It will allow to highligt the best results in
                ### terms of accuracy. If ACCU is false, the comparisonCriteria will be AUC. It will allow to highlight
                ### the best results in terms of AUC.
                if (ACCU):
                    comparisonCriteria = np.mean(accuTrain)
                else:
                    comparisonCriteria = np.mean(AUCsTrain)

                ### We keep the best intermediary (degree) results in terms of Accuracy or AUC (depending on ACCU)
                if (comparisonCriteria > interBESTPERF):
                    fileN = fileName + '---BestIndividual' + '.npy'
                    newGenCounter = 0
                    interBESTPERF, interbestAucT, interbestAucStd, interbestSensT, interbestSensStdT, interbestSpeT, interbestSpeStdT, interbestAccT, interbestAccStdT, interbestMccT, interbestMccStdT, interbestSens, interbestSpe, interbestAcc, interbestMcc, interbestAuc, interangleType, internbFeatures, interfileName, internumgen, intertiming, interangleTheta = keepBestPerf(
                        comparisonCriteria, AUCsTrain, sensitivitiesTrain, specificitiesTrain, accuTrain, mccTrain,
                        finalSensitivity, finalSpecificity, finalAccu, finalMCC, finalAUC, degrees, population, fitness,
                        fileN, numGeneration, diffDate, THETA)

                ### We keep the best final results in terms of Accuracy or AUC (depending on ACCU)
                if (comparisonCriteria > BESTPERF):
                    fileN=interfileName
                    BESTPERF, bestAucT, bestAucStd, bestSensT, bestSensStdT, bestSpeT, bestSpeStdT, bestAccT, bestAccStdT, bestMccT, bestMccStdT, bestSens, bestSpe, bestAcc, bestMcc, bestAuc, angleType, nbFeatures, bestFileName, numgen, timing, angleTheta = keepBestPerf(
                        comparisonCriteria, AUCsTrain, sensitivitiesTrain, specificitiesTrain, accuTrain, mccTrain,
                        finalSensitivity, finalSpecificity, finalAccu, finalMCC, finalAUC, degrees, population, fitness,
                        fileN, numGeneration, diffDate, THETA)

                ### Generation number incrementation
                numGeneration = numGeneration + 1
                newGenCounter = newGenCounter + 1

                ### If the generation number reach the MAXGEN parameter we gave in input, the program stops.
                if (numGeneration > MAXGEN):
                    break
                ### If there were no better generation find in the last 50, the program stops.
                if (newGenCounter == 50):
                    print("Interruption, last generation >50 ")
                    break

            ### Displays (and keep into the results file), the best results obtain during the degree processing
            print("\n\nResult for degree type:" + str(degrees))
            FinalDisplay(interbestAucT, interbestSensT, interbestSensStdT,
                         interbestSpeT, interbestSpeStdT, interbestAccT, interbestAccStdT, interbestMccT,
                         interbestMccStdT, interbestSens, interbestSpe, interbestAcc, interbestMcc, interbestAuc,
                         interangleType, internbFeatures, interfileName, internumgen, intertiming, interangleTheta,
                         TRAIN_RATIO)
            print("Degree process time:" + str(datetime.now() - start_looptime))

            ### Appends the list for the scatter final graphs of Acc, AUC and nb of features
            accTabGraph.append(interbestAccT)
            aucTabGraph.append(interbestAucT)
            nbVarTabGraph.append(float(internbFeatures))
            degTabGraph.append(math.degrees(angle))
            ### End degree

        ### Appends the list of the best observed results during all runs.
        accTab.append(bestAcc)
        aucTab.append(bestAucT)
        nbVarTab.append(float(nbFeatures))

        ### Displays (and keep into the results file), the best results obtain during all runs
        print("\n\nBest recorded accuracy result:")
        FinalDisplay(bestAucT, bestSensT, bestSensStdT, bestSpeT, bestSpeStdT, bestAccT,
                     bestAccStdT, bestMccT, bestMccStdT, bestSens, bestSpe, bestAcc, bestMcc, bestAuc, angleType,
                     nbFeatures, bestFileName, numgen, timing, angleTheta, TRAIN_RATIO)
        print("Run exploration time:" + str(datetime.now() - start_BeforeLooptime))

        ### End iteration

    stopTime = datetime.now()

    ### Activates to save final performances in .csv file
    saveFinalPerf(PATHRES, DATASET, accTabGraph, aucTabGraph, nbVarTabGraph)
    ###

    ### Displays the final statistics of Min, Max and averages.
    DisplayStats(accTabGraph, nbVarTabGraph, aucTabGraph, degTabGraph, stopTime, start_BeforeLooptime)

    ### Displays the table of used seeds for each run
    DiplaySeeds(seedsTab)

    ### Displays the name of selected features for the best obtained results
    DisplaySelectedFeatures(DATASET_NAME, bestFileName, TOTAL_NBR_FEATURES)

    ### Traces all scatter graphs of accuracy, AUC. Gives tables.
    TraceGraph(nbVarTabGraph, accTabGraph, "Accuracy", "blue", DATASET)
    TraceGraph(nbVarTabGraph, aucTabGraph, "AUC", "red", DATASET)
    TraceGraph(nbVarTabGraph, degTabGraph, "Angle", "black", DATASET)
    generateTable(degTabGraph, nbVarTabGraph, accTabGraph, aucTabGraph, DATASET)

    if (RES_FILE != "none"):
        sys.stdout = orig_stdout
        f.close()

    ### End experiment


main()

print("THAT'S ALL FOLKS !!!")
