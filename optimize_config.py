from lib2to3.pytree import convert
from modules.helper import *
from modules.preprocessing import * 
import os
import time
from prettytable import PrettyTable
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
    stressUnit = "MPa"
    convertUnit = 1
elif CPLaw == "DB":
    stressUnit = "Pa"
    convertUnit = 1e-6

optimizerName = dataConfig["optimizerName"] 

yieldingPoints = {
    "PH": 
    {
        "linear_uniaxial_RD": 0.004,
        "nonlinear_biaxial_RD": 0.118,
        "nonlinear_biaxial_TD": 0.118,
        "nonlinear_planestrain_RD": 0.092,
        "nonlinear_planestrain_TD": 0.092,
        "nonlinear_uniaxial_RD": 0.061, 
        "nonlinear_uniaxial_TD": 0.061,
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

curveIndices = dataConfig["curveIndices"] 

if isinstance(curveIndices, int):
    curveIndices = [curveIndices]
else:
    curveIndices = [int(index) for index in curveIndices.split(";")] 

expTypes = dataConfig["expTypes"] 

if len(expTypes) == 1:
    expTypes = [expTypes]
else:
    expTypes = [expType for expType in expTypes.split(";")] 

numberOfCurves = len(curveIndices)

searchingSpace = dataConfig["searchingSpace"]  

searchingType = dataConfig["searchingType"]
 
roundContinuousDecimals = dataConfig["roundContinuousDecimals"]  

largeLinearYieldingDevGlobal = dataConfig["largeLinearYieldingDev"]

smallLinearYieldingDevGlobal = dataConfig["smallLinearYieldingDev"]

largeLinearHardeningDevGlobal = dataConfig["largeLinearHardeningDev"]  

smallLinearHardeningDevGlobal = dataConfig["smallLinearHardeningDev"]  

largeNonlinearYieldingDevGlobal = dataConfig["largeNonlinearYieldingDev"]

smallNonlinearYieldingDevGlobal = dataConfig["smallNonlinearYieldingDev"]

largeNonlinearHardeningDevGlobal = dataConfig["largeNonlinearHardeningDev"]  

smallNonlinearHardeningDevGlobal = dataConfig["smallNonlinearHardeningDev"]  

# Setting the weights of the two yield stress objective functions:  
weightsYielding = {"wy1": 0.0995, "wy2": 0.0005}

# Setting the weights of the two hardening objective functions:  
weightsHardening = {"wh1": 0.099, "wh2": 0.001}

# Setting the weights of the seven loadings in the yielding and hardening fitness function:  

# weightsLoading = {
#     "linear_uniaxial_RD": 0.19, 
#     "nonlinear_biaxial_RD": 0.08, 
#     "nonlinear_biaxial_TD": 0.16,     
#     "nonlinear_planestrain_RD": 0.04,     
#     "nonlinear_planestrain_TD": 0.12,     
#     "nonlinear_uniaxial_RD": 0.01, 
#     "nonlinear_uniaxial_TD": 0.4,
# }

weightsLoading = {
    "linear_uniaxial_RD": 0.19, 
    "nonlinear_biaxial_RD": 0.06, 
    "nonlinear_biaxial_TD": 0.17,     
    "nonlinear_planestrain_RD": 0.04,     
    "nonlinear_planestrain_TD": 0.15,     
    "nonlinear_uniaxial_RD": 0.01, 
    "nonlinear_uniaxial_TD": 0.38,
}

# weightsLoading = {
#     "linear_uniaxial_RD": 0.22, 
#     "nonlinear_biaxial_RD": 0.05, 
#     "nonlinear_biaxial_TD": 0.18,     
#     "nonlinear_planestrain_RD": 0.03,     
#     "nonlinear_planestrain_TD": 0.16,     
#     "nonlinear_uniaxial_RD": 0.01, 
#     "nonlinear_uniaxial_TD": 0.35,
# }

paramsFormatted = {
    "PH": {
        "a": "a", 
        "gdot0": "γ̇₀", 
        "h0": "h₀", 
        "n": "n", 
        "tau0": "τ₀", 
        "tausat": "τₛₐₜ",
        "self": "self", 
        "coplanar": "coplanar", 
        "collinear": "collinear", 
        "orthogonal": "orthogonal", 
        "glissile": "glissile", 
        "sessile": "sessile", 
    },
    "DB": {
        "dipole": "d̂α", 
        "islip": "iₛₗᵢₚ", 
        "omega": "Ω", 
        "p": "p", 
        "q": "q", 
        "tausol": "τₛₒₗ",
        "Qs": "Qs",
        "Qc": "Qc",
        "v0": "v₀",
        "rho_e": "ρe",
        "rho_d": "ρd",   
    },
}

paramsUnit = {
    "PH": {
        "a": "", # Empty string means this parameter is dimensionless
        "gdot0": "s⁻¹", 
        "h0": "MPa", 
        "n": "", 
        "tau0": "MPa", 
        "tausat": "MPa",
        "self": "", 
        "coplanar": "", 
        "collinear": "", 
        "orthogonal": "", 
        "glissile": "", 
        "sessile": "", 
    },
    "DB": {
        "dipole": "b", 
        "islip": "", 
        "omega": "b³", 
        "p": "", 
        "q": "", 
        "tausol": "MPa",
        "Qs": "J",
        "Qc": "J",
        "v0": "m/s",
        "rho_e": "m⁻²",
        "rho_d": "m⁻²",   
    },
}

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

if CPLaw == "PH":
    law = "phenomenological law"
elif CPLaw == "DB":
    law = "dislocation-based law"

target_curves = [f"{CPLaw}{index}" for index in curveIndices]

largeLinearYieldingDevGlobalPercent = f"{largeLinearYieldingDevGlobal}%"

smallLinearYieldingDevGlobalPercent = f"{smallLinearYieldingDevGlobal}%"

largeLinearHardeningDevGlobalPercent = f"{largeLinearHardeningDevGlobal}%"

smallLinearHardeningDevGlobalPercent = f"{smallLinearHardeningDevGlobal}%"

largeNonlinearYieldingDevGlobalPercent = f"{largeNonlinearYieldingDevGlobal}%"

smallNonlinearYieldingDevGlobalPercent = f"{smallNonlinearYieldingDevGlobal}%"

largeNonlinearHardeningDevGlobalPercent = f"{largeNonlinearHardeningDevGlobal}%"

smallNonlinearHardeningDevGlobalPercent = f"{smallNonlinearHardeningDevGlobal}%"

configMessages = [  
    f"\nWelcome to the Crystal Plasticity Parameter Calibration software\n\n",
    f"The configurations you have chosen: \n",       
]

logTable = PrettyTable()

logTable.field_names = ["Configurations", "User choice"]
logTable.add_row(["Server", server])
logTable.add_row(["Hyperqueue", hyperqueue])
logTable.add_row(["Initial simulation number", initialSims])
logTable.add_row(["Initial simulation method", method])
logTable.add_row(["Searching space", searchingSpace])
logTable.add_row(["Searching type", searchingType])
logTable.add_row(["Material", material])
logTable.add_row(["CP Law", CPLaw])
logTable.add_row(["Optimizer", optimizerName])
logTable.add_row(["Target curves", ', '.join(target_curves)])
logTable.add_row(["Experimental types", ', '.join(expTypes)])
logTable.add_row(["Stress unit", stressUnit])
logTable.add_row(["Rounding decimals", roundContinuousDecimals])
logTable.add_row(["Large linear yielding dev", largeLinearYieldingDevGlobalPercent])
logTable.add_row(["Small linear yielding dev", smallLinearYieldingDevGlobalPercent])
logTable.add_row(["Large linear hardening dev", largeLinearHardeningDevGlobalPercent])
logTable.add_row(["Small linear hardening dev", smallLinearHardeningDevGlobalPercent])
logTable.add_row(["Large nonlinear yielding dev", largeNonlinearYieldingDevGlobalPercent])
logTable.add_row(["Small nonlinear yielding dev", smallNonlinearYieldingDevGlobalPercent])
logTable.add_row(["Large nonlinear hardening dev", largeNonlinearHardeningDevGlobalPercent])
logTable.add_row(["Small nonlinear hardening dev", smallNonlinearHardeningDevGlobalPercent])

configMessages.append(logTable.get_string())
configMessages.append("\n\n")

#########################################################
# Creating necessary directories for the configurations #
#########################################################

# Adding one more string at the start fso expTypes[curveIndex] will return the exact exp type
# Since curveIndices start counting from 1
expTypes.insert(0, "Dummy to make exp type count at 1")

def checkCreate(path):
    if not os.path.exists(path):
        os.mkdir(path)

# For log
checkCreate("log")
path = f"log/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")

# For manualParams
checkCreate("manualParams")
path = f"manualParams/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")

# For parameter_analysis
checkCreate("parameter_analysis")
path = f"parameter_analysis/{material}"
checkCreate(path)
checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")

# For results 
checkCreate("results")
path = f"results/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for curveIndex in curveIndices:
    checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}")
checkCreate(f"{path}/{CPLaw}/universal")


# For simulations

checkCreate("simulations")
path = f"simulations/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for curveIndex in curveIndices:
    checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}")
    for loading in loadings:
        checkCreate(f"{path}/{CPLaw}/{CPLaw}{curveIndex}_{optimizerName}_{searchingSpace}/{loading}")
checkCreate(f"{path}/{CPLaw}/universal")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/universal/{loading}")  

# For targets 
checkCreate("targets")
path = f"targets/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")
checkCreate(f"{path}/{CPLaw}/param_info")

# For templates
checkCreate("templates")
path = f"templates/{material}"
checkCreate(path)

checkCreate(f"{path}/{CPLaw}")
for loading in loadings:
    checkCreate(f"{path}/{CPLaw}/{loading}")
checkCreate(f"{path}/{CPLaw}/param_info")



