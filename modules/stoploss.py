import numpy as np
from math import *
from sklearn.metrics import mean_squared_error
import time

def isMonotonic(sim_stress):
    if len(turningStressPoints(sim_stress)) == 0:
        return True
    else:
        return False 

# The strain range of the Bauschinger effect
BauschingerRange = 0.03
BauschingerWeight = 0

def getIndexBeforeStrainLevel(strain, level):
    for i in range(len(strain)):
        if strain[i] > level:
            return i - 1

# Linearly increasing weight for MSE. If this is 0, then it is constant weight
linearStep = 0

#########################################################################
# Yield stress objective and fitness functions for one and all loadings #
#########################################################################

def Y1Linear(exp_stress, sim_stress, interpolating_strain):
    expYieldStress = exp_stress[1]
    simYieldStress = sim_stress[1] 
    return abs(expYieldStress - simYieldStress) 

def Y2Linear(exp_stress, sim_stress, interpolating_strain):
    expSlope = (exp_stress[2] - exp_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    simSlope = (sim_stress[2] - sim_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    return abs(expSlope - simSlope) 

def fitnessYieldingLinear(exp_stress, sim_stress, interpolating_strain, weightsYielding):
    wy1 = weightsYielding["wy1"]
    wy2 = weightsYielding["wy2"]
    if not isMonotonic(sim_stress):
        return 10e12
    else:
        #print(wy1 * Y1Linear(exp_stress, sim_stress, interpolating_strain))
        #print(wy2 * Y2Linear(exp_stress, sim_stress, interpolating_strain))
        #time.sleep(2)
        return (wy1 * Y1Linear(exp_stress, sim_stress, interpolating_strain) + wy2 * Y2Linear(exp_stress, sim_stress, interpolating_strain))

def Y1Nonlinear(exp_stress, sim_stress, interpolating_strain):
    expYieldStress = exp_stress[1]
    simYieldStress = sim_stress[1] 
    return (expYieldStress - simYieldStress) ** 2

def Y2Nonlinear(exp_stress, sim_stress, interpolating_strain):
    expSlope = (exp_stress[2] - exp_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    simSlope = (sim_stress[2] - sim_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    return (expSlope - simSlope) ** 2

def fitnessYieldingNonlinear(exp_stress, sim_stress, interpolating_strain, weightsYielding):
    wy1 = weightsYielding["wy1"]
    wy2 = weightsYielding["wy2"]
    
    return (wy1 * Y1Nonlinear(exp_stress, sim_stress, interpolating_strain) + wy2 * Y2Nonlinear(exp_stress, sim_stress, interpolating_strain))

def fitnessYieldingAllLoadings(exp_curves, sim_curves, loadings, weightsLoading, weightsYielding):
    fitnessAllLoadings = 0
    for loading in loadings:
        if loading == "linear_uniaxial_RD":
            fitnessAllLoadings += weightsLoading[loading] * fitnessYieldingLinear(exp_curves[loading]["stress"], sim_curves[loading]["stress"], exp_curves[loading]["strain"], weightsYielding)
        else: 
            fitnessAllLoadings += weightsLoading[loading] * fitnessYieldingNonlinear(exp_curves[loading]["stress"], sim_curves[loading]["stress"], exp_curves[loading]["strain"], weightsYielding)
    return fitnessAllLoadings

##############################################################################
# Hardening objective and fitness functions for one loading and all loadings #
##############################################################################

def H1Linear(exp_stress, sim_stress, interpolating_strain):
    linearlyIncreasingWeights = [] 
    counter = 0
    for _ in range(len(interpolating_strain)):
        linearlyIncreasingWeights.append(1 + counter)
        counter += linearStep
    weighted_MSE = np.average((exp_stress - sim_stress) ** 2, weights=linearlyIncreasingWeights)
    return weighted_MSE

def H2Linear(exp_stress, sim_stress, interpolating_strain): 
    linearlyIncreasingWeights = [] 
    counter = 0
    for _ in range(len(interpolating_strain) - 1):
        linearlyIncreasingWeights.append(1 + counter)
        counter += linearStep

    diffStrain = np.diff(interpolating_strain)
    exp_stress_difference = np.diff(exp_stress)/diffStrain  
    sim_stress_difference = np.diff(sim_stress)/diffStrain
    weighted_MSE = np.average((exp_stress_difference - sim_stress_difference) ** 2, weights=linearlyIncreasingWeights)
    return weighted_MSE

def fitnessHardeningLinear(exp_stress, sim_stress, interpolating_strain, weightsHardening):
    wh1 = weightsHardening["wh1"]
    wh2 = weightsHardening["wh2"]
    #weighted_wh1 = wh1*H1Linear(exp_stress, sim_stress, interpolating_strain)
    #weighted_wh2 = wh2*H2Linear(exp_stress, sim_stress, interpolating_strain)
    #print(weighted_wh1)
    #print(weighted_wh2)
    if not isMonotonic(sim_stress):
        return 10e12
    else:
        # return (wh1*H1Linear(exp_stress, sim_stress, interpolating_strain) + wh2*H2Linear(exp_stress, sim_stress, interpolating_strain))
        return wh2 * H2Linear(exp_stress, sim_stress, interpolating_strain)

def H1Nonlinear(exp_stress, sim_stress, interpolating_strain):

    BauschingerIndex = getIndexBeforeStrainLevel(interpolating_strain, interpolating_strain[0] + BauschingerRange)

    linearlyIncreasingWeights = [] 
    for _ in range(0, BauschingerIndex + 1):
        linearlyIncreasingWeights.append(BauschingerWeight)
    counter = 0
    for _ in range(BauschingerIndex + 1, len(interpolating_strain)):
        linearlyIncreasingWeights.append(1 + counter)
        counter += linearStep
    weighted_MSE = np.average((exp_stress - sim_stress) ** 2, weights=linearlyIncreasingWeights)
    return weighted_MSE
    
def H2Nonlinear(exp_stress, sim_stress, interpolating_strain): 
    BauschingerIndex = getIndexBeforeStrainLevel(interpolating_strain, interpolating_strain[0] + BauschingerRange)
    linearlyIncreasingWeights = [] 
    
    for _ in range(0, BauschingerIndex + 1):
        linearlyIncreasingWeights.append(BauschingerWeight)
    counter = 0
    for _ in range(BauschingerIndex + 1, len(interpolating_strain) - 1):
        linearlyIncreasingWeights.append(1)
        counter += linearStep
    
    diffStrain = np.diff(interpolating_strain)
    exp_stress_difference = np.diff(exp_stress)/diffStrain  
    sim_stress_difference = np.diff(sim_stress)/diffStrain
    weighted_MSE = np.average((exp_stress_difference - sim_stress_difference)**2, weights=linearlyIncreasingWeights)
    return weighted_MSE

def fitnessHardeningNonlinear(exp_stress, sim_stress, interpolating_strain, weightsHardening):
    wh1 = weightsHardening["wh1"]
    wh2 = weightsHardening["wh2"]
    #weighted_wh1 = wh1*H1Nonlinear(exp_stress, sim_stress, interpolating_strain)
    #weighted_wh2 = wh2*H2Nonlinear(exp_stress, sim_stress, interpolating_strain)
    #print(weighted_wh1)
    #print(weighted_wh2)
    return (wh1*H1Nonlinear(exp_stress, sim_stress, interpolating_strain) + wh2*H2Nonlinear(exp_stress, sim_stress, interpolating_strain))

def fitnessHardeningAllLoadings(exp_curves, sim_curves, loadings, weightsLoading, weightsHardening):
    fitnessAllLoadings = 0
    for loading in loadings:
        #print(loading)
        if loading == "linear_uniaxial_RD":
            #print(weightsLoading[loading] * fitnessHardeningLinear(exp_curves[loading]["stress"], sim_curves[loading]["stress"], exp_curves[loading]["strain"], weightsHardening))
            fitnessAllLoadings += weightsLoading[loading] * fitnessHardeningLinear(exp_curves[loading]["stress"], sim_curves[loading]["stress"], exp_curves[loading]["strain"], weightsHardening)
        else:
            #print(weightsLoading[loading] * fitnessHardeningNonlinear(exp_curves[loading]["stress"], sim_curves[loading]["stress"], exp_curves[loading]["strain"], weightsHardening))
            fitnessAllLoadings += weightsLoading[loading] * fitnessHardeningNonlinear(exp_curves[loading]["stress"], sim_curves[loading]["stress"], exp_curves[loading]["strain"], weightsHardening)
        #time.sleep(2)
    #time.sleep(180)
    return fitnessAllLoadings

################################################################
# Stopping criteria functions for one loading and all loadings #
################################################################

def insideYieldingDevLinear(exp_stress, sim_stress, interpolating_strain, percentDeviation):
    expYieldStress = exp_stress[1]
    simYieldStress = sim_stress[1] 
    upper = expYieldStress * (1 + percentDeviation * 0.01) 
    lower = expYieldStress * (1 - percentDeviation * 0.01) 
    if simYieldStress >= lower and simYieldStress <= upper and isMonotonic(sim_stress):
        return True
    else:
        return False

# This function is actually not used in this project
def insideYieldingDevNonlinear(exp_stress, sim_stress, interpolating_strain, percentDeviation):
    expYieldStress = exp_stress[1]
    simYieldStress = sim_stress[1] 
    upper = expYieldStress * (1 + percentDeviation * 0.01) 
    lower = expYieldStress * (1 - percentDeviation * 0.01) 
    if simYieldStress >= lower and simYieldStress <= upper:
        return True
    else:
        return False

def insideYieldingDevAllLoadings(exp_curves, sim_curves, loadings, percentDeviations):
    notSatisfiedLoadings = []
    allLoadingsSatisfied = True
    for loading in loadings:
        expStress = exp_curves[loading]['stress']
        simStress = sim_curves[loading]['stress']
        interpolating_strain = exp_curves[loading]['strain']
        percentDeviation = percentDeviations[loading]
        if loading == "linear_uniaxial_RD":
            thisLoadingSatisfied = insideYieldingDevLinear(expStress, simStress, interpolating_strain, percentDeviation)
        else:
            thisLoadingSatisfied = insideYieldingDevNonlinear(expStress, simStress, interpolating_strain, percentDeviation)
        allLoadingsSatisfied = allLoadingsSatisfied and thisLoadingSatisfied
        if not thisLoadingSatisfied:
            notSatisfiedLoadings.append(loading)
    return (allLoadingsSatisfied, notSatisfiedLoadings)

def turningStressPoints(trueStress):
    differences = np.diff(trueStress)
    index = 1
    turningIndices = []
    while index < differences.size:
        if (differences[index - 1] <= 0 and differences[index] >= 0) or (differences[index - 1] >= 0 and differences[index] <= 0):
            turningIndices.append(index)
        index += 1
    return turningIndices

def insideHardeningDevLinear(exp_stress, sim_stress, interpolating_strain, percentDeviation):
    upperStress = exp_stress * (1 + percentDeviation * 0.01) 
    lowerStress = exp_stress * (1 - percentDeviation * 0.01) 
    if not isMonotonic(sim_stress):
        return False
    for i in range(exp_stress.size):
        if sim_stress[i] < lowerStress[i] or sim_stress[i] > upperStress[i]:
            return False 
    return True

def insideHardeningDevNonlinear(exp_stress, sim_stress, interpolating_strain, percentDeviation):
    BauschingerIndex = getIndexBeforeStrainLevel(interpolating_strain, interpolating_strain[0] + BauschingerRange)

    pruned_exp_stress = exp_stress[BauschingerIndex:]
    pruned_sim_stress = sim_stress[BauschingerIndex:]

    pruned_upperStress = pruned_exp_stress * (1 + percentDeviation * 0.01) 
    pruned_lowerStress = pruned_exp_stress * (1 - percentDeviation * 0.01) 
    if not isMonotonic(sim_stress):
        return False
    for i in range(pruned_exp_stress.size):
        if pruned_sim_stress[i] < pruned_lowerStress[i] or pruned_sim_stress[i] > pruned_upperStress[i]:
            return False 
    return True

def insideHardeningDevAllLoadings(exp_curves, sim_curves, loadings, percentDeviations):
    notSatisfiedLoadings = []
    allLoadingsSatisfied = True
    for loading in loadings:
        expStress = exp_curves[loading]['stress']
        simStress = sim_curves[loading]['stress']
        interpolating_strain = exp_curves[loading]['strain']
        percentDeviation = percentDeviations[loading]
        if loading == "linear_uniaxial_RD":
            thisLoadingSatisfied = insideHardeningDevLinear(expStress, simStress, interpolating_strain, percentDeviation)
        else:
            thisLoadingSatisfied = insideHardeningDevNonlinear(expStress, simStress, interpolating_strain, percentDeviation)
        allLoadingsSatisfied = allLoadingsSatisfied and thisLoadingSatisfied
        if not thisLoadingSatisfied:
            notSatisfiedLoadings.append(loading)
    return (allLoadingsSatisfied, notSatisfiedLoadings)

