import ast
import math

import numpy as np



def computeAngle(u, v):
    """
    Angle computation
    :param u
    :param v
    :return: np.arccos(scalprod / (normU * normV)) * 180 / np.pi
    :rtype: float
    """
    scalprod = np.vdot(u, v)
    normU = np.linalg.norm(u)
    normV = np.linalg.norm(v)
    return np.arccos(scalprod / (normU * normV)) * 180 / np.pi



def coodX(theta):
    """
    YX Compass coordinate computation
    :param theta
    :return: r * math.cos(theta)
    :rtype: float
    """
    r = 1.4142135623730951;
    return r * math.cos(theta)



def coodY(theta):
    """
    Y Compass coordinate computation
    :param theta
    :return: r * math.sin(theta)
    :rtype: float
    """

    r = 1.4142135623730951;
    return r * math.sin(theta)



def sumPred(tab):
    """
    Sums items of tab
    :param tab
    :return: sum
    :rtype: float or integer
    """
    sum = 0
    for i in range(len(tab)):
        sum += tab[i]

    return sum


#############################################################################
def int2binaire(number, numberSize):
    #############################################################################
    """
    will create a binary number from a starting number
    we want the result in an array of n = size Number of boxes
    :param number
    :param numberSize
    :return: tab
    :rtype:list
    
    """
    stringVar = bin(number)
    stringVar = stringVar[2:]

    # complete with 0 in front if the stringVar is shorter than numberSize
    while (len(stringVar) < numberSize):
        stringVar = "0" + stringVar

    tab = np.zeros((numberSize, 1))

    for i in range(numberSize):  # convert string to array
        if stringVar[i] == "1":
            tab[i] = 1

    return tab


#############################################################################
def nombreDeUns(tab):
    #############################################################################
    """
    count the number of 1 in an array
    :param tan
    :return:sum(tab)
    :rtype:integer
    """
    return sum(tab)


#############################################################################
def calculRacineCarreeSensSpec(sensitivity, specificity):
    #############################################################################
    """
    compute the square root of (1-sensitivity) * (1-sensitivity) + (1-specificity) * (1-specificity)
    :param sensitivity
    :param specificity
    :return: np.sqrt((1 - sensitivity) * (1 - sensitivity) + (1 - specificity) * (1 - specificity))
    :rtype: float
    """
    return np.sqrt((1 - sensitivity) * (1 - sensitivity) + (1 - specificity) * (1 - specificity))


def setMAXDEG(EXPLORE_TYPE, RANGE):
    """
    set the angle upper bound limit one can reach from a starting angle depending on the chosen exploration type.
    :param degrees
    :param EXPLORE_TYPE
    :param MAX_DEG
    :param RANGE
    :return: MAX_DEG
    :rtype:integer
    """
    MAX_DEG = -100
    if EXPLORE_TYPE == "quarter":
        MAX_DEG = 46
    elif EXPLORE_TYPE == "low" or EXPLORE_TYPE == "high":
        MAX_DEG = 8
    elif EXPLORE_TYPE == "very low" or EXPLORE_TYPE == "very high" or EXPLORE_TYPE == "mid-high" or EXPLORE_TYPE == "mid-low":
        MAX_DEG = 6
    elif EXPLORE_TYPE == "mid":
        MAX_DEG = 24
    elif EXPLORE_TYPE == "partial":
        MAX_DEG = 8
    elif EXPLORE_TYPE == "range":
        RANGE = ast.literal_eval(RANGE)
        MAX_DEG = RANGE[1] - RANGE[0] + 1
    elif EXPLORE_TYPE == "range-micro":
        RANGE = ast.literal_eval(RANGE)
        MAX_DEG = (RANGE[1] - RANGE[0]) * 2 + 1
    else:
        print("EXPLORE TYPE NOT RECOGNIZED")

    return MAX_DEG


def setAngle(degrees, EXPLORE_TYPE, MAX_DEG, RANGE):
    """
    possible values are "range", "range-micro","quarter","low","high","very low", "very high", "mid-high", "mid-low",
    "mid", "partial".
    - "range" allows to explore with a step of 1 degree whereas "range-micro" allows to scan with a step of 0.5 degree ;
    - "quarter" allows to explore solutions from 0 to 90 with a step of 1 degree ;
    - "partial" allows to explore 9 typical angles (11pi/24, 5pi/12, 3pi/8, pi/3, pi/4, pi/6, pi/8, pi/12, pi/24) ;
    - parameters like e.g., low, mid or high are preset parameters that allow to explore solution in a limited part of
    the compass (e.g.,"low" explores small angles compass solution whereas "high" explores high angles compass solution.
    "mid" explores balanced solution between objectives).

    :param degrees
    :param EXPLORE_TYPE
    :param MAX_DEG
    :param RANGE
    :return: angle
    :rtype:float

    """

    angle = -100
    if EXPLORE_TYPE == "quarter":
        if degrees < 1:
            angle = 0.01
        elif degrees >= MAX_DEG - 1:
            angle = 1.555
        else:
            angle = math.radians(degrees * 2)

    elif EXPLORE_TYPE == "low":
        if degrees < 1:
            angle = 0.01
        elif degrees >= MAX_DEG - 1:
            angle = math.radians(15)
        else:
            angle = math.radians(degrees * 2)
    elif EXPLORE_TYPE == "mid-low":
        if degrees < 1:
            angle = math.radians(5)
        else:
            angle = math.radians(degrees * 2 + 5)
    elif EXPLORE_TYPE == "very low":
        if degrees < 1:
            angle = 0.01
        elif degrees >= MAX_DEG - 1:
            angle = math.radians(10)
        else:
            angle = math.radians(degrees * 2)

    elif EXPLORE_TYPE == "mid":
        if degrees < 1:
            angle = math.radians(22.5)
        elif degrees >= MAX_DEG - 1:
            angle = math.radians(67.5)
        else:
            angle = math.radians(degrees * 2 + 22.5)

    elif EXPLORE_TYPE == "mid-high":
        if degrees < 1:
            angle = math.radians(76)
        else:
            angle = math.radians(degrees * 2 + 76)

    elif EXPLORE_TYPE == "high":
        if degrees < 1:
            angle = math.radians(76)
        elif degrees >= MAX_DEG - 1:
            angle = 1.555
        else:
            angle = math.radians(degrees * 2 + 76)

    elif EXPLORE_TYPE == "very high":
        if degrees < 1:
            angle = math.radians(80)
        elif degrees >= MAX_DEG - 1:
            angle = 1.555
        else:
            angle = math.radians(degrees * 2 + 80)

    elif EXPLORE_TYPE == "range":
        RANGE = ast.literal_eval(RANGE)
        if degrees < 1 and RANGE[0] != 0:
            angle = math.radians(RANGE[0])
        elif degrees < 1 and RANGE[0] == 0:
            angle = 0.01
        elif degrees + RANGE[0] >= 90:
            angle = 1.555
        else:
            angle = math.radians(degrees + RANGE[0])

    elif EXPLORE_TYPE == "range-micro":
        RANGE = ast.literal_eval(RANGE)
        if degrees < 1 and RANGE[0] != 0:
            angle = math.radians(RANGE[0])
        elif degrees < 1 and RANGE[0] == 0:
            angle = 0.01
        elif degrees + RANGE[0] >= 180:
            angle = 1.555
        else:
            angle = math.radians(degrees / 2 + RANGE[0])

    elif EXPLORE_TYPE == "partial":
        if degrees == 0:
            angle = np.pi / 24
        elif degrees == 1:
            angle = np.pi / 12
        elif degrees == 2:
            angle = np.pi / 8
        elif degrees == 3:
            angle = np.pi / 6
        elif degrees == 4:
            angle = np.pi / 4
        elif degrees == 5:
            angle = np.pi / 3
        elif degrees == 6:
            angle = 3 * np.pi / 8
        elif degrees == 7:
            angle = 5 * np.pi / 12
        elif degrees == 8:
            angle = 11 * np.pi / 24
    else:
        print("EXPLORE TYPE NOT RECOGNIZED")

    return angle


def keepBestPerf(comparisonCriteria, AUCsTrain, sensitvitiesTrain, specificitiesTrain, accuTrain, mccTrain,
                 finalSensitivity, finalSpecificity, finalAccu, finalMCC, finalAUC, degrees, population, fitness, fileN,
                 numGeneration,
                 diffDate, THETA):
    """
    Computes different metrics for best identified results and return it
    :param comparisonCriteria
    :param AUCsTrain
    :param sensitvitiesTrain:
    :param specificitiesTrain
    :param accuTrain
    :param mccTrain
    :param finalSensitivity
    :param finalSpecificity:
    :param finalAccu
    :param finalMCC
    :param finalAUC
    :param degrees
    :param population
    :param fitness
    :param fileN
    :param numGeneration
    :param diffDate
    :param THETA
    :return: bpf, baucT, baucSt, bSensT, bSensSt, bSpeT, bSpeSt, bAccT, bAccSt, bMccT, bMccSt, bSens, bSpe, bAcc, bMcc, bAuc, angleTy, nbF, fN, nGen, timing, angleTH
    :rtype: any
    """

    bpf = comparisonCriteria
    baucT = np.mean(AUCsTrain)
    baucSt = np.round(np.std(AUCsTrain), 3)
    bSensT = np.round(np.mean(sensitvitiesTrain) * 100, 3)
    bSensSt = np.round(np.std(sensitvitiesTrain) * 100, 2)
    bSpeT = np.round(np.mean(specificitiesTrain) * 100, 3)
    bSpeSt = np.round(np.std(specificitiesTrain) * 100, 3)
    bAccT = np.round(np.mean(accuTrain), 3)
    bAccSt = np.round(np.std(accuTrain), 3)
    bMccT = np.round(np.mean(mccTrain), 3)
    bMccSt = np.round(np.std(mccTrain), 3)
    bSens = np.round(finalSensitivity, 3) * 100
    bSpe = np.round(finalSpecificity, 3) * 100
    bAcc = np.round(finalAccu, 3)
    bMcc = np.round(finalMCC, 3)
    bAuc = np.round(finalAUC, 3)
    angleTy = degrees
    nbF = str(sum(population[np.argmax(fitness)]))
    fN = fileN
    nGen = numGeneration
    timing = diffDate
    angleTH = THETA

    return bpf, baucT, baucSt, bSensT, bSensSt, bSpeT, bSpeSt, bAccT, bAccSt, bMccT, bMccSt, bSens, bSpe, bAcc, bMcc, bAuc, angleTy, nbF, \
           fN, nGen, timing, angleTH
