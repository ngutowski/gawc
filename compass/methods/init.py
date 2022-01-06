# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:31:24 2021

@author: n.gutowski
"""

import json


def initialize():
    """
    Initialize all metrics to 0
    :param: None
    :return: BESTPERF, bestAucT, bestAucStd, bestSensT, bestSensStdT, bestSpeT, bestSpeStdT, bestAccT, bestAccStdT, bestMccT, bestMccStdT, bestSens, bestSpe, bestAcc, bestMcc, bestAuc, angle, nbFeatures, fileName
    :rtype: any

    """


    BESTPERF = 0
    bestAucT = 0
    bestAucStd = 0
    bestSensT = 0
    bestSensStdT = 0
    bestSpeT = 0
    bestSpeStdT = 0
    bestAccT = 0
    bestAccStdT = 0
    bestMccT = 0
    bestMccStdT = 0
    bestSens = 0
    bestSpe = 0
    bestAcc = 0
    bestMcc = 0
    bestAuc = 0
    angle = 0
    nbFeatures = ""
    fileName = ""

    return BESTPERF, bestAucT, bestAucStd, bestSensT, bestSensStdT, bestSpeT, bestSpeStdT, bestAccT, bestAccStdT, bestMccT, bestMccStdT, bestSens, bestSpe, bestAcc, bestMcc, bestAuc, angle, nbFeatures, fileName


def parameters(jsonfile):
    """
    Load parameters from JSON parameters file
    :param jsonfile
    :return: CROSS_VALIDATIONS_NUMBER, NORMALIZATION, DATASET, ML_METHOD, INDIVIDUALS_NUMBER, INDIVIDUAL_INJECTION, DIMENSION_MODE, ACCU, NB_RUN, PATHRES, TRAIN_RATIO, nomDataset, MAXGEN, NBFEAT_RATIO, RES_FILE, EXPL_TYPE, RANGE,SEED
    :rtype:any
    """

    with open(jsonfile, 'r') as f:
        params = json.load(f)

    CROSS_VALIDATIONS_NUMBER = params['nbCV']

    NORMALIZATION = False
    if (params['norm'] == "true"):
        NORMALIZATION = True

    ML_METHOD = params['mlMethod']
    DATASET = params['datasetRes']
    INDIVIDUALS_NUMBER = params['nbIndiv']

    INDIVIDUAL_INJECTION = False
    if (params['injIndiv'] == "true"):
        INDIVIDUAL_INJECTION = True

    DIMENSION_MODE = params['dimMode']
    ACCU = False
    if (params['acc'] == "true"):
        ACCU = True

    NB_RUN = params['nbRun']

    TRAIN_RATIO = params['trainRatio']
    nomDataset = params['datasetName']

    MAXGEN = params['maxgen']

    PATHRES = "/results/"

    NBFEAT_RATIO = params['nbFeatRatio']

    RES_FILE = params['resFile']

    EXPL_TYPE = params['exploreType']
    RANGE = params['range']
    SEED = params['seed']

    return CROSS_VALIDATIONS_NUMBER, NORMALIZATION, DATASET, ML_METHOD, INDIVIDUALS_NUMBER, INDIVIDUAL_INJECTION, DIMENSION_MODE, ACCU, NB_RUN, PATHRES, TRAIN_RATIO, nomDataset, MAXGEN, NBFEAT_RATIO, RES_FILE, EXPL_TYPE, RANGE,SEED


def initializeFinalMetrics():
    """
    Initialize Final metrics
    :param: None
    :return:accTab, aucTab, nbFeatTab, seedsTab, accTabGraph, aucTabGraph, nbFeatTabGraph, degTabGraph
    :rtype:list
    """
    accTab = []
    aucTab = []
    nbFeatTab = []
    seedsTab = []
    accTabGraph = []
    aucTabGraph = []
    nbFeatTabGraph = []
    degTabGraph = []
    return accTab, aucTab, nbFeatTab, seedsTab, accTabGraph, aucTabGraph, nbFeatTabGraph, degTabGraph
