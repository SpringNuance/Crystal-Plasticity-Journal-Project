import numpy as np
import random


param_range = {'dipole': {'low': 1.0, 'high': 50.0, 'step': 1.0, 'default': 5, 'round': 0}, 'islip': {'low': 1.0, 'high': 100.0, 'step': 5.0, 'default': 40, 'round': 0}, 'omega': {'low': 1.0, 'high': 50.0, 'step': 1.0, 'default': 5, 'round': 0}, 'p': {'low': 0.05, 'high': 1.0, 'step': 0.05, 'default': '-', 'round': 2}, 'q': {'low': 1.0, 'high': 2.0, 'step': 0.05, 'default': '-', 'round': 2}, 'tausol': {'low': 0.01, 'high': 
1.0, 'step': 0.01, 'default': '-', 'round': 2}}

def universalSim(param_range, initialSims, rounding):
    linspaceValues = {}
    for param in param_range:
        linspaceValues[param] = np.linspace(param_range[param]["low"], param_range[param]["high"], num = initialSims)
        linspaceValues[param] = linspaceValues[param].tolist()   
    universalParams = []
    print(linspaceValues)
    for _ in range(initialSims):
        paramSet = []
        for param in linspaceValues:
            random.shuffle(linspaceValues[param])
            paramSet.append(round(linspaceValues[param].pop(), rounding))
        universalParams.append(tuple(paramSet))
    return universalParams

for param in universalSim(param_range, 5, 6):
    print(param)

