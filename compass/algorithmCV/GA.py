# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:34:52 2021

@author: n.gutowski
"""

import math
import random as rnd

import numpy as np

from compass.algorithmCV.crossval import computeSensitivitiesSpecificitiesAUCsAccus
from compass.methods.utilMethods import computeAngle


#############################################################################
def featuresSelection(selectedFeatures, x_train):
    #############################################################################
    """
        will select in x_train the right columns according to the selected features

        :param selectedFeatures
        :param x_train

        :return: x_train_res
        :rtype: list

    """
    numberofColumns_x_train_res = 0

    for i in range(len(selectedFeatures)):
        if selectedFeatures[i] == 1:
            if numberofColumns_x_train_res == 0:
                numberofColumns_x_train_res = numberofColumns_x_train_res + 1
                x_train_res = x_train[:, i]
            else:
                numberofColumns_x_train_res = numberofColumns_x_train_res + 1
                x_train_res = np.column_stack((x_train_res, x_train[:, i]))

    return x_train_res


#############################################################################
def featuresandFinalsSelection(stringNumberBinary, x_train, x_test_final):
    #############################################################################

    """
        will select in x_train and x_test_final the good columns according to stringNumberBinary
        :param stringNumberBinary
        :param x_train
        :param x_test_final
        :return: x_train_res, x_test_final_res
        :rtype: list
    """
    numberofColumns_x_train_res = 0

    for i in range(len(stringNumberBinary)):
        if stringNumberBinary[i] == 1:
            if numberofColumns_x_train_res == 0:
                numberofColumns_x_train_res = numberofColumns_x_train_res + 1
                x_train_res = x_train[:, i]
                x_test_final_res = x_test_final[:, i]
            else:
                numberofColumns_x_train_res = numberofColumns_x_train_res + 1
                x_train_res = np.column_stack((x_train_res, x_train[:, i]))
                x_test_final_res = np.column_stack((x_test_final_res, x_test_final[:, i]))

    return x_train_res, x_test_final_res


#############################################################################
def initPopulation(INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION, x_train, y_train, individual_injection,
                   CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE, NBFEAT_RATIO):
    #############################################################################

    """ initialization of the population of individuals + init of fitness at -1

        one thus finds oneself with a matrix of NOMBRE_INDIVIDUS x DIMENSION_INDIVIDU
                    for example 128 individuals of 300 variables

       + 1 fitness column matrix

       :param INDIVIDUALS_NUMBER
       :param INDIVIDUAL_DIMENSION
       :param x_train
       :param y_train
       :param individual_injection
       :param CROSS_VALIDATIONS_NUMBER
       :param ML_METHOD
       :param COMPASS
       :param ACCU
       :param DIMENSION_MODE
       :param NBFEAT_RATIO
       :return: population, fitness
       :rtype:list

    """
    population = np.random.randint(2, size=(INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION))
    fitness = np.zeros((INDIVIDUALS_NUMBER, 1)) - 1

    for i in range(INDIVIDUAL_DIMENSION):  # create INDIVIDUAL_DIMENSION with from n = 1 to n = INDIVIDUAL_DIMENSION "1"
        for j in range(INDIVIDUAL_DIMENSION):  # in short: upper left diagonal of "1"
            if (i + j <= INDIVIDUAL_DIMENSION):
                population[i, j] = 1
            else:
                population[i, j] = 0

    for i in range(INDIVIDUAL_DIMENSION):
        population[i, 0] = 1

    if individual_injection:
        population[0] = np.load('LastIndividual.npy')

    for i in range(INDIVIDUALS_NUMBER):
        fitness[i] = computeFitness(population[i], x_train, y_train, CROSS_VALIDATIONS_NUMBER,
                                    ML_METHOD, COMPASS, ACCU, DIMENSION_MODE, NBFEAT_RATIO)

    return population, fitness


#############################################################################
def startAgainFromAPopulation(fileName, NUMBER_OF_INDIVIDUALS, INDIVIDUAL_DIMENSION,
                              x_train, y_train, CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE,
                              NBFEAT_RATIO):
    #############################################################################

    """
    Restart from a population
    :param fileName
    :param NUMBER_OF_INDIVIDUALS
    :param INDIVIDUAL_DIMENSION
    :param x_train
    :param y_train
    :param CROSS_VALIDATIONS_NUMBER
    :param ML_METHOD
    :param ACCU
    :param DIMENSION_MODE
    :param NBFEAT_RATIO
    :return: population, fitness
    :rtype:list
    """

    population = np.load(fileName)
    fitness = np.zeros((NUMBER_OF_INDIVIDUALS, 1)) - 1
    for i in range(NUMBER_OF_INDIVIDUALS):
        fitness[i] = computeFitness(population[i], x_train, y_train, CROSS_VALIDATIONS_NUMBER,
                                    ML_METHOD, COMPASS, ACCU, DIMENSION_MODE, NBFEAT_RATIO)
    return population, fitness


#############################################################################
def tournamentSelection(population, fitness, INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION):
    #############################################################################

    """ returns the result of the selection by tournament.

        So, we start from NOMBRE_INDIVIDUS and we end up with NOMBRE_INDIVIDUS / 2

        + 1/2 of the associated column matrix of fitness

        :param population
        :param fitness
        :param INDIVIDUALS_NUMBER
        :param INDIVIDUAL_DIMENSION
        :return: resPopulation, resFitness
        :rtype: list

    """
    resPopulation = np.zeros((round(INDIVIDUALS_NUMBER / 2), INDIVIDUAL_DIMENSION))
    resFitness = np.zeros((round(INDIVIDUALS_NUMBER / 2), 1))

    i1 = 0
    i2 = 0

    while (i1 < INDIVIDUALS_NUMBER):
        if (fitness[i1] > fitness[i1 + 1]):
            resPopulation[i2, :] = population[i1]
            resFitness[i2] = fitness[i1]

        else:
            resPopulation[i2, :] = population[i1 + 1]
            resFitness[i2] = fitness[i1 + 1]

        i1 = i1 + 2
        i2 = i2 + 1

    return resPopulation, resFitness


#############################################################################
def crossing2Parents(parent1, parent2, INDIVIDUAL_DIMENSION):
    ###########################################################################

    """ returns the cross of 2 parents

       two-point crossover

       :param parent1
       :param parent2
       :param INDIVIDUAL_DIMENSION
       :return: child1, child2
       :rtype:list

    """
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)

    draw1 = rnd.randint(0, INDIVIDUAL_DIMENSION - 1)
    draw2 = rnd.randint(0, INDIVIDUAL_DIMENSION - 1)

    cutoutPoint1 = min(draw1, draw2)
    cutoutPoint2 = max(draw1, draw2)

    # invert the 2 parts between the cutting points

    for i in range(cutoutPoint1, cutoutPoint2):
        temp = child1[i]
        child1[i] = child2[i]
        child2[i] = temp

    return child1, child2


#############################################################################
def computeFitness(individual, x_train, y_train, CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE,
                   NBFEAT_RATIO):
    #############################################################################

    """
        Computation of an individual's fitness

        :param individual
        :param x_train
        :param y_train
        :param CROSS_VALIDATIONS_NUMBER
        :param ML_METHOD
        :param COMPASS
        :param ACCU
        :param DIMENSION_MODE
        :param NBFEAT_RATIO

        :return: fitness
        :rtype: float
    """

    if (sum(individual) < 2):  # less than two features lets the problem becomes useless. Hence, no need to continue.
        fitness = -1
    else:
        selected_x_train = featuresSelection(individual, x_train)
        sensitivitiesTrain, specificitiesTrain, aucsTrain, accuTrain, mccTrain = computeSensitivitiesSpecificitiesAUCsAccus(
            CROSS_VALIDATIONS_NUMBER, selected_x_train, y_train, ML_METHOD)

        ### 3D Compass fitness computation case. Possible perspectives : code more possibilities 2D, 4D,... ,nD
        ### or/and with other criteria.
        scoreMax = 1 - (2 / (x_train.shape[1] / NBFEAT_RATIO))
        featureScore = (1 - (sum(individual) / (x_train.shape[1] / NBFEAT_RATIO))) / scoreMax

        score_acc = np.mean(accuTrain) / 100
        score_auc = (np.mean(aucsTrain) - 0.5) * 2
        oVector = np.array([featureScore, score_acc, score_auc])

        alpha = math.radians(computeAngle(oVector, COMPASS))
        fitness = np.linalg.norm(oVector) * math.cos(alpha)

    return fitness


#############################################################################
def crossingallParents(parentsPopulation, fitnessParents, INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION,
                       x_train, y_train, CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE,
                       NBFEAT_RATIO):
    #############################################################################

    """
        Returns the result of all the parents to be crossed, so a population
        of NOMBRE_INDIVIDUS / 2

        + fitness

        :param parentsPopulation
        :param fitnessParents
        :param INDIVIDUALS_NUMBER
        :param INDIVIDUAL_DIMENSION
        :param x_train
        :param y_train
        :param CROSS_VALIDATIONS_NUMBER
        :param ML_METHOD
        :param COMPASS
        :param ACCU
        :param DIMENSION_MODE
        :param NBFEAT_RATIO

        :return: resPopulation, resFitness
        :rtype: list
    """

    resPopulation = np.zeros((round(INDIVIDUALS_NUMBER / 2), INDIVIDUAL_DIMENSION))
    resFitness = np.zeros((round(INDIVIDUALS_NUMBER / 2), 1))

    i = 0
    while (i < round(INDIVIDUALS_NUMBER / 2)):
        child1, child2 = crossing2Parents(parentsPopulation[i], parentsPopulation[i + 1], INDIVIDUAL_DIMENSION)
        resPopulation[i] = child1
        resPopulation[i + 1] = child2

        resFitness[i] = computeFitness(child1, x_train, y_train, CROSS_VALIDATIONS_NUMBER,
                                       ML_METHOD, COMPASS, ACCU, DIMENSION_MODE, NBFEAT_RATIO)
        resFitness[i + 1] = computeFitness(child2, x_train, y_train,
                                           CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE,
                                           NBFEAT_RATIO)
        i = i + 2

    return resPopulation, resFitness


#############################################################################
def mutation(population, fitness, INDIVIDUALS_NUMBER, INDIVIDUAL_DIMENSION, rate,
             indexNotToMutate, x_train, y_train, CROSS_VALIDATIONS_NUMBER, ML_METHOD, COMPASS, ACCU, DIMENSION_MODE,
             NBFEAT_RATIO):
    #############################################################################

    """ will mutate the population of all individuals with a certain rate. Example of mutation with a rate of 50%:
    we would mutate 1 over 2 individuals of the population. For example, here we would transfer 64 individuals out of
    the 128.
    Of course, we do not touch the best individual that we have found so far
     (individual referenced by the index:indexNotToMutate)

         Modifies the population and fitness tables.

         :param population
         :param fitness
         :param INDIVIDUALS_NUMBER
         :param INDIVIDUAL_DIMENSION
         :param rate
         :param indexNotToMutate
         :param x_train
         :param y_train
         :param CROSS_VALIDATIONS_NUMBER
         :param ML_METHOD
         :param COMPASS
         :param ACCU
         :param DIMENSION_MODE
         :param NBFEAT_RATIO

         :return: resPopulation, resFitness
         :rtype: list


    """
    resPopulation = np.copy(population)
    resFitness = np.copy(fitness)

    for i in range(INDIVIDUALS_NUMBER):
        draw = rnd.randint(0, 100)
        if ((draw < rate) and (i != indexNotToMutate)):
            draw1 = rnd.randint(0, INDIVIDUAL_DIMENSION - 1)
            draw2 = rnd.randint(0, INDIVIDUAL_DIMENSION - 1)

            temp = resPopulation[i, draw1]
            resPopulation[i, draw1] = resPopulation[i, draw2]
            resPopulation[i, draw2] = temp

            resFitness[i] = computeFitness(resPopulation[i], x_train, y_train, CROSS_VALIDATIONS_NUMBER,
                                           ML_METHOD, COMPASS, ACCU, DIMENSION_MODE, NBFEAT_RATIO)
    return resPopulation, resFitness
