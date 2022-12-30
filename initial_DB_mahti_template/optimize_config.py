from lib2to3.pytree import convert
from modules.helper import *
from modules.preprocessing import * 
import os
import time

###########################################################
#                                                         #
#         CRYSTAL PLASTICITY PARAMETER CALIBRATION        #
#   Tools required: DAMASK and Finnish Supercomputer CSC  #
#                                                         #
###########################################################

# -------------------------------------------------------------------
#   Stage 0: Choose the CP model, the optimization algorithm, number of initial simulations,
#   the target curve index to fit, project path folder and material name
# -------------------------------------------------------------------

dataConfig = pd.read_excel("optimize_config.xlsx", nrows= 1, engine="openpyxl")
dataConfig = dataConfig.T.to_dict()[0]
# print(dataConfig)

hyperqueue = dataConfig["hyperqueue"]

material = dataConfig["material"]

server = dataConfig["server"]

CPLaw = dataConfig["CPLaw"]
if CPLaw == "PH":
    numberOfParams = 4
    stressUnit = "MPa"
    convertUnit = 1
elif CPLaw == "DB":
    numberOfParams = 6
    stressUnit = "Pa"
    convertUnit = 1e-6

optimizerName = dataConfig["optimizerName"] 

yieldingPoints = {
    "PH": 
    {
        "linear_uniaxial_RD": 0.004,
        "nonlinear_biaxial_RD": 0.118,
        "nonlinear_biaxial_TD": 0.118,
        "nonlinear_planestrain_RD": 0.091,
        "nonlinear_planestrain_TD": 0.091,
        "nonlinear_uniaxial_RD": 0.060, 
        "nonlinear_uniaxial_TD": 0.060,
    },
    
    "DB": 
    {
        "linear_uniaxial_RD": 0.004,
        "nonlinear_biaxial_RD": 0.118,
        "nonlinear_biaxial_TD": 0.118,
        "nonlinear_planestrain_RD": 0.091,
        "nonlinear_planestrain_TD": 0.091,
        "nonlinear_uniaxial_RD": 0.060, 
        "nonlinear_uniaxial_TD": 0.060,
    },
}

initialSims = dataConfig["initialSims"]  

method = dataConfig["method"]  

numberOfExistingCurves = dataConfig["numberOfExistingCurves"]  

curveIndices = dataConfig["curveIndices"] 
if curveIndices == "all":
    curveIndices = [*range(1, numberOfExistingCurves + 1)]
elif isinstance(curveIndices, int):
    curveIndices = [curveIndices]
else:
    curveIndices = [int(index) for index in curveIndices.split(";")] 

numberOfCurves = len(curveIndices)

searchingSpace = dataConfig["searchingSpace"]  

searchingType = dataConfig["searchingType"]
 
roundContinuousDecimals = dataConfig["roundContinuousDecimals"]  

yieldStressDev = dataConfig["yieldStressDev"]

hardeningDev = dataConfig["hardeningDev"]  

# Setting the weights of the two yield stress objective functions:  
weightsYield = {"wy1": 0.999, "wy2": 0.001}

# Setting the weights of the four hardening objective functions:  
weightsHardening = {"wh1": 0.9, "wh2": 0.025, "wh3": 0.05, "wh4": 0.025}

# The 7 types of loadings: 1 linear loading and 6 nonlinear loadings
loadings = ["linear_uniaxial_RD", 
            "nonlinear_biaxial_RD", 
            "nonlinear_biaxial_TD",     
            "nonlinear_planestrain_RD",     
            "nonlinear_planestrain_TD",     
            "nonlinear_uniaxial_RD", 
            "nonlinear_uniaxial_TD"]

# The project path folder
projectPath = os.getcwd()

###############################################
#  Printing the configurations to the console #
###############################################
print("\nWelcome to the Crystal Plasticity Parameter Calibration software")
print("\nThe configurations you have chosen: ")

print("\nThe material is", material)
if CPLaw == "PH":
    law = "phenomenological law"
elif CPLaw == "DB":
    law = "dislocation-based law"
print("\nThe stress unit is", stressUnit)
print("\nThe CP Law is", law)
print("\nThe optimization algorithm is", optimizerName)
print("\nThe server is", server)
print("\nHyperqueue mode is used:", hyperqueue)
print("\nThe number of initial simulations", initialSims)
print("\nThe initial simulation method is", method)
print("\nThe number of existing target curves is", numberOfExistingCurves)
target_curves = [f"{CPLaw}{index}" for index in curveIndices]
print("\nThe target curves for optimization are", target_curves)
print("\nThe searching space is", searchingSpace)
if searchingSpace == "continuous":
    print("\nThe rounding decimals is", roundContinuousDecimals)
yieldStressDevPercent = f"{yieldStressDev}%"
print("\nThe yield stress deviation percentage is", yieldStressDevPercent)
hardeningDevPercent = f"{hardeningDev}%"
print("\nThe hardening deviation percentage is", hardeningDevPercent)
print("\nThe path to your main project folder is")
print(projectPath)


#########################################################
# Creating necessary directories for the configurations #
#########################################################

'''
def checkCreate(path):
    if not os.path.exists(path):
        os.mkdir(path)


# For log
path = f"log/{material}"

checkCreate(f"{path}/{CPLaw}")

# For manualParams
path = f"manualParams/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")

# For parameter_analysis
path = f"parameter_analysis/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")

# For results 
path = f"results/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for curveIndex in curveIndices:
    checkCreate(f"{path}/{CPLaw}/{optimizerName}{curveIndex}")
    for loading in loadings:
        checkCreate(f"{path}/{CPLaw}/{optimizerName}{curveIndex}/{loading}")
checkCreate(f"{path}/{CPLaw}/universal")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/universal/{loading}")  

# For simulations
path = f"simulations/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for curveIndex in curveIndices:
    checkCreate(f"{path}/{CPLaw}/{optimizerName}{curveIndex}")
    for loading in loadings:
        checkCreate(f"{path}/{CPLaw}/{optimizerName}{curveIndex}/{loading}")
checkCreate(f"{path}/{CPLaw}/universal")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/universal/{loading}")  

# For targets 
path = f"targets/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")
checkCreate(f"{path}/{CPLaw}/param_info")

# For templates
path = f"templates/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")
checkCreate(f"{path}/{CPLaw}/param_info")

print("\nGenerating necessary directories")

'''