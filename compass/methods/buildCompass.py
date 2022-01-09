# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:58:51 2021

@author: n.gutowski
"""

import numpy as np

from compass.methods.utilMethods import coodX, coodY, computeAngle


def buildCompass(DIMENSION_MODE, angle):
    """
    Method that allow to build the Compass. This is the 3D Compass method proposed in AI in Medicine journal
    :param DIMENSION_MODE
    :param angle
    :return: COMPASS, THETA, targetedAUC
    :rtype: list or float
    """
    if DIMENSION_MODE == "3D":
        ## AUC objective should always be the maximum since we look for the best classification quality

        targetedAUC = 1

        ## Compute the AUC objective corresponding score

        AUCCOOD = (targetedAUC - 0.5) * 2

        ## Compute the Number of features objective corresponding score

        FEATCOOD = coodX(angle)

        ## Compute the AUC objective corresponding score

        ACCUCOOD = coodY(angle)
        axisAbs = np.array([1, 0])

        ## Compute the angle THETA of the compass

        THETA = computeAngle(axisAbs, np.array([FEATCOOD, ACCUCOOD]))

        ## Compute the Compass as a vector represented by a 3D array

        COMPASS = np.array([FEATCOOD, ACCUCOOD, AUCCOOD])

    return COMPASS, THETA, targetedAUC
