import numpy as np
from math import *

################################################
# Yield stress objective and fitness functions #
################################################
def Y1(exp_stress, sim_stress):
    expYieldStress = exp_stress[1]
    simYieldStress = sim_stress[1] 
    return abs(expYieldStress - simYieldStress)

def Y2(exp_stress, sim_stress, interpolating_strain):
    expSlope = (exp_stress[2] - exp_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    simSlope = (sim_stress[2] - sim_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    return abs(expSlope - simSlope)

def fitness_yield(exp_stress, sim_stress, interpolating_strain, wy1, wy2):
    return (wy1 * Y1(exp_stress, sim_stress) + wy2 * Y2(exp_stress, sim_stress, interpolating_strain))

#############################################
# Hardening objective and fitness functions #
#############################################
def H1(exp_stress, sim_stress): 
    return np.sqrt(np.sum(np.square(exp_stress - sim_stress))/np.sum(np.square(exp_stress)))

def H2(exp_stress, sim_stress, interpolating_strain): 
    exp_stress_d1 = np.diff(exp_stress)/np.diff(interpolating_strain)  
    sim_stress_d1 = np.diff(sim_stress)/np.diff(interpolating_strain)
    return np.sqrt(np.sum(np.square(sim_stress_d1 - exp_stress_d1))/np.sum(np.square(exp_stress_d1)))

def H3(exp_stress, sim_stress):
    return np.max(np.sqrt(np.square(exp_stress - sim_stress)/sum(np.square(exp_stress))))

def H4(exp_stress, sim_stress, interpolating_strain):
    exp_stress_d1 = np.diff(exp_stress)/np.diff(interpolating_strain)
    sim_stress_d1 = np.diff(sim_stress)/np.diff(interpolating_strain)
    return np.max(np.sqrt(np.square(sim_stress_d1 - exp_stress_d1)/np.sum(np.square(exp_stress_d1))))

def fitness_hardening(exp_stress, sim_stress, interpolating_strain, wh1, wh2, wh3, wh4):
    return ( wh1*H1(exp_stress, sim_stress) + wh2*H2(exp_stress, sim_stress, interpolating_strain) 
            + wh3*H3(exp_stress, sim_stress) + wh4*H4(exp_stress, sim_stress, interpolating_strain))

###############################
# Stopping criteria functions #
###############################

def insideYieldStressDev(exp_stress, sim_stress, percentDeviation):
    expYieldStress = exp_stress[0]
    simYieldStress = sim_stress[0] 
    upper = expYieldStress * (1 + percentDeviation * 0.01) 
    lower = expYieldStress * (1 - percentDeviation * 0.01) 
    if simYieldStress >= lower and simYieldStress <= upper:
        return True
    else:
        return False

def insideHardeningDev(exp_target, sim_stress, percentDeviation):
    upperStress = exp_target * (1 + percentDeviation * 0.01) 
    lowerStress = exp_target * (1 - percentDeviation * 0.01) 
    for i in range(exp_target.size):
        if sim_stress[i] < lowerStress[i] or sim_stress[i] > upperStress[i]:
            return False 
    return True

def havingAtMostIntersections(exp_target, sim_stress, numberOfAllowedIntersections):
    def getSign(stress):
        if stress >= 0:
            sign = 1
        else: 
            sign = -1
        return sign     
    numberOfIntersections = 0
    difference = exp_target - sim_stress
    sign = getSign(difference[0])
    for i in range(1, difference.size):
        tempSign = getSign(difference[i])
        if sign != tempSign:
            numberOfIntersections += 1
            sign = tempSign
    if numberOfIntersections <= numberOfAllowedIntersections:
        return True
    else:
        return False

def numberOfIntersections(exp_target, sim_stress):
    def getSign(stress):
        if stress >= 0:
            sign = 1
        else: 
            sign = -1
        return sign     
    numberOfIntersections = 0
    difference = exp_target - sim_stress
    sign = getSign(difference[0])
    for i in range(1, difference.size):
        tempSign = getSign(difference[i])
        if sign != tempSign:
            numberOfIntersections += 1
            sign = tempSign
    return numberOfIntersections