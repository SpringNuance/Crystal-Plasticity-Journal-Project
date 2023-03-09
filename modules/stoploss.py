import numpy as np
from math import *
from sklearn.metrics import mean_squared_error
import time


def turningStressPoints(trueStress):
    differences = np.diff(trueStress)
    index = 1
    turningIndices = []
    while index < differences.size:
        if (differences[index - 1] <= 0 and differences[index] >= 0) or (differences[index - 1] >= 0 and differences[index] <= 0):
            turningIndices.append(index)
        index += 1
    return turningIndices

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

############################################################################################
# Yield stress objective and loss functions for one and all loadings for linear loading #
############################################################################################

################################################################
# There is no need of yielding functions for nonlinear loading #
################################################################

def Y1Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain):
    expYieldStress = interpolate_exp_stress[1]
    simYieldStress = interpolate_sim_stress[1] 
    return abs(expYieldStress - simYieldStress) 

def Y2Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain):
    expSlope = (interpolate_exp_stress[2] - interpolate_exp_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    simSlope = (interpolate_sim_stress[2] - interpolate_sim_stress[0]) /(interpolating_strain[2] - interpolating_strain[0])
    return abs(1 - abs(expSlope/simSlope)) 

def lossYieldingOneLinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain, weightsYieldingConstitutive):
    wy1 = weightsYieldingConstitutive["wy1"]
    wy2 = weightsYieldingConstitutive["wy2"]
    if not isMonotonic(interpolate_sim_stress):
        return 10e12
    else:
        weightedY1Linear = wy1 * Y1Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain)
        weightedY2Linear = wy2 * Y2Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain)
        #print(weightedY1Linear)
        #print(weightedY2Linear)
        #time.sleep(2)
        return weightedY1Linear + weightedY2Linear

def lossYieldingAllLinear(interpolate_exp_curve, interpolate_sim_curve, loadings, weightsYieldingLinearLoadings, weightsYieldingConstitutive):
    lossAllLoadings = 0
    for loading in loadings:
        if loading.startswith("linear"):
            lossAllLoadings += weightsYieldingLinearLoadings[loading] * lossYieldingOneLinear(interpolate_exp_curve[loading]["stress"], interpolate_sim_curve[loading]["stress"], interpolate_exp_curve[loading]["strain"], weightsYieldingConstitutive)
    return lossAllLoadings

#################################################################################################
# Hardening objective and loss functions for one loading and all loadings for linear loading #
#################################################################################################

def H1Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain):
    linearlyIncreasingWeights = [] 
    counter = 0
    for _ in range(len(interpolating_strain)):
        linearlyIncreasingWeights.append(1 + counter)
        counter += linearStep
    weighted_MSE = np.average((interpolate_exp_stress - interpolate_sim_stress) ** 2, weights=linearlyIncreasingWeights)
    return weighted_MSE

def H2Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain): 
    linearlyIncreasingWeights = [] 
    counter = 0
    for _ in range(len(interpolating_strain) - 1):
        linearlyIncreasingWeights.append(1 + counter)
        counter += linearStep

    diffStrain = np.diff(interpolating_strain)
    exp_stress_difference = np.diff(interpolate_exp_stress)/diffStrain  
    sim_stress_difference = np.diff(interpolate_sim_stress)/diffStrain
    weighted_MSE = np.average((exp_stress_difference - sim_stress_difference) ** 2, weights=linearlyIncreasingWeights)
    return weighted_MSE

def lossHardeningOneLinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain, weightsHardeningConstitutive):   
    wh2 = weightsHardeningConstitutive["wh2"]
    weightedH2Linear = wh2 * H2Linear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain)
    #print(weightedH2Linear)
    if not isMonotonic(interpolate_sim_stress):
        return 10e12
    else:
        return weightedH2Linear

def lossHardeningAllLinear(interpolate_exp_curve, interpolate_sim_curve, loadings, weightsHardeningLinearLoadings, weightsHardeningConstitutive):
    lossAllLoadings = 0
    for loading in loadings:
        if loading.startswith("linear"):
            lossAllLoadings += weightsHardeningLinearLoadings[loading] * lossHardeningOneLinear(interpolate_exp_curve[loading]["stress"], interpolate_sim_curve[loading]["stress"], interpolate_exp_curve[loading]["strain"], weightsHardeningConstitutive)
    return lossAllLoadings

####################################################################################################
# Hardening objective and loss functions for one loading and all loadings for nonlinear loading #
####################################################################################################

def H1Nonlinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain):

    BauschingerIndex = getIndexBeforeStrainLevel(interpolating_strain, interpolating_strain[0] + BauschingerRange)

    linearlyIncreasingWeights = [] 
    for _ in range(0, BauschingerIndex + 1):
        linearlyIncreasingWeights.append(BauschingerWeight)
    counter = 0
    for _ in range(BauschingerIndex + 1, len(interpolating_strain)):
        linearlyIncreasingWeights.append(1 + counter)
        counter += linearStep
    weighted_MSE = np.average((interpolate_exp_stress - interpolate_sim_stress) ** 2, weights=linearlyIncreasingWeights)
    return weighted_MSE
    
def H2Nonlinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain): 
    BauschingerIndex = getIndexBeforeStrainLevel(interpolating_strain, interpolating_strain[0] + BauschingerRange)
    linearlyIncreasingWeights = [] 
    
    for _ in range(0, BauschingerIndex + 1):
        linearlyIncreasingWeights.append(BauschingerWeight)
    counter = 0
    for _ in range(BauschingerIndex + 1, len(interpolating_strain) - 1):
        linearlyIncreasingWeights.append(1)
        counter += linearStep
    
    diffStrain = np.diff(interpolating_strain)
    exp_stress_difference = np.diff(interpolate_exp_stress)/diffStrain  
    sim_stress_difference = np.diff(interpolate_sim_stress)/diffStrain
    weighted_MSE = np.average((exp_stress_difference - sim_stress_difference)**2, weights=linearlyIncreasingWeights)
    return weighted_MSE

def lossHardeningOneNonlinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain, weightsHardeningConstitutive):
    wh1 = weightsHardeningConstitutive["wh1"]
    wh2 = weightsHardeningConstitutive["wh2"]
    weightedH1Nonlinear = wh1*H1Nonlinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain)
    weightedH2Nonlinear = wh2*H2Nonlinear(interpolate_exp_stress, interpolate_sim_stress, interpolating_strain)
    #print(weightedH1Nonlinear)
    #print(weightedH2Nonlinear)
    return weightedH1Nonlinear + weightedH2Nonlinear

#########################################################
# Hardening loss for all loadings, linear and nonlinear #
#########################################################

def lossHardeningAllLoadings(interpolate_exp_curve, interpolate_sim_curve, loadings, weightsHardeningAllLoadings, weightsHardeningConstitutive):
    lossAllLoadings = 0
    for loading in loadings:
        #print(loading)
        if loading.startswith("linear"):
            weightedHardeningLoading = weightsHardeningAllLoadings[loading] * lossHardeningOneLinear(interpolate_exp_curve[loading]["stress"], interpolate_sim_curve[loading]["stress"], interpolate_exp_curve[loading]["strain"], weightsHardeningConstitutive)
            #print(weightedHardeningLoading)
            lossAllLoadings += weightedHardeningLoading
        else:
            weightsHardeningLoading = weightsHardeningAllLoadings[loading] * lossHardeningOneNonlinear(interpolate_exp_curve[loading]["stress"], interpolate_sim_curve[loading]["stress"], interpolate_exp_curve[loading]["strain"], weightsHardeningConstitutive)
            #print(weightedHardeningLoading)
            lossAllLoadings += weightsHardeningLoading
        #time.sleep(2)
    #time.sleep(180)
    return lossAllLoadings

################################################################
# Stopping criteria functions for one loading and all loadings #
################################################################

def insideYieldingDevLinear(exp_stress, sim_stress, interpolating_strain, deviationPercent):
    expYieldStress = exp_stress[1]
    simYieldStress = sim_stress[1] 
    upper = expYieldStress * (1 + deviationPercent * 0.01) 
    lower = expYieldStress * (1 - deviationPercent * 0.01) 
    if simYieldStress >= lower and simYieldStress <= upper and isMonotonic(sim_stress):
        return True
    else:
        return False


def insideYieldingDevAllLinear(exp_curves, sim_curves, loadings, deviationPercent):
    notSatisfiedLoadings = []
    allLoadingsSatisfied = True
    for loading in loadings:
        expStress = exp_curves[loading]['stress']
        simStress = sim_curves[loading]['stress']
        interpolating_strain = exp_curves[loading]['strain']
        if loading.startswith("linear"):
            thisLoadingSatisfied = insideYieldingDevLinear(expStress, simStress, interpolating_strain, deviationPercent)
            allLoadingsSatisfied = allLoadingsSatisfied and thisLoadingSatisfied
            if not thisLoadingSatisfied:
                notSatisfiedLoadings.append(loading)
    return (allLoadingsSatisfied, notSatisfiedLoadings)

def insideHardeningDevLinear(exp_stress, sim_stress, interpolating_strain, deviationPercent):
    upperStress = exp_stress * (1 + deviationPercent * 0.01) 
    lowerStress = exp_stress * (1 - deviationPercent * 0.01) 
    if not isMonotonic(sim_stress):
        return False
    for i in range(exp_stress.size):
        if sim_stress[i] < lowerStress[i] or sim_stress[i] > upperStress[i]:
            return False 
    return True

def insideHardeningDevNonlinear(exp_stress, sim_stress, interpolating_strain, deviationPercent):
    BauschingerIndex = getIndexBeforeStrainLevel(interpolating_strain, interpolating_strain[0] + BauschingerRange)

    pruned_exp_stress = exp_stress[BauschingerIndex:]
    pruned_sim_stress = sim_stress[BauschingerIndex:]

    pruned_upperStress = pruned_exp_stress * (1 + deviationPercent * 0.01) 
    pruned_lowerStress = pruned_exp_stress * (1 - deviationPercent * 0.01) 
    if not isMonotonic(sim_stress):
        return False
    for i in range(pruned_exp_stress.size):
        if pruned_sim_stress[i] < pruned_lowerStress[i] or pruned_sim_stress[i] > pruned_upperStress[i]:
            return False 
    return True

def insideHardeningDevAllLinear(exp_curves, sim_curves, loadings, deviationPercent):
    notSatisfiedLoadings = []
    allLoadingsSatisfied = True
    for loading in loadings:
        expStress = exp_curves[loading]['stress']
        simStress = sim_curves[loading]['stress']
        interpolating_strain = exp_curves[loading]['strain']
        if loading.startswith("linear"):
            thisLoadingSatisfied = insideHardeningDevLinear(expStress, simStress, interpolating_strain, deviationPercent)
            allLoadingsSatisfied = allLoadingsSatisfied and thisLoadingSatisfied
            if not thisLoadingSatisfied:
                notSatisfiedLoadings.append(loading)
    return (allLoadingsSatisfied, notSatisfiedLoadings)

def insideHardeningDevAllNonlinear(exp_curves, sim_curves, loadings, deviationPercent):
    notSatisfiedLoadings = []
    allLoadingsSatisfied = True
    for loading in loadings:
        expStress = exp_curves[loading]['stress']
        simStress = sim_curves[loading]['stress']
        interpolating_strain = exp_curves[loading]['strain']
        if loading.startswith("nonlinear"):
            thisLoadingSatisfied = insideHardeningDevNonlinear(expStress, simStress, interpolating_strain, deviationPercent)
            allLoadingsSatisfied = allLoadingsSatisfied and thisLoadingSatisfied
            if not thisLoadingSatisfied:
                notSatisfiedLoadings.append(loading)
    return (allLoadingsSatisfied, notSatisfiedLoadings)

def insideHardeningDevAllLoadings(exp_curves, sim_curves, loadings, deviationPercent):
    notSatisfiedLoadings = []
    allLoadingsSatisfied = True
    for loading in loadings:
        expStress = exp_curves[loading]['stress']
        simStress = sim_curves[loading]['stress']
        interpolating_strain = exp_curves[loading]['strain']
        if loading.startswith("linear"):
            thisLoadingSatisfied = insideHardeningDevLinear(expStress, simStress, interpolating_strain, deviationPercent)
        else:
            thisLoadingSatisfied = insideHardeningDevNonlinear(expStress, simStress, interpolating_strain, deviationPercent)
        allLoadingsSatisfied = allLoadingsSatisfied and thisLoadingSatisfied
        if not thisLoadingSatisfied:
            notSatisfiedLoadings.append(loading)
    return (allLoadingsSatisfied, notSatisfiedLoadings)

