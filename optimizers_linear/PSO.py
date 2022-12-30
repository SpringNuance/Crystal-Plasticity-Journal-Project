from modules.stoploss import *
from modules.helper import *
import bayes_opt
import time
from sklearn.metrics import mean_squared_error
from math import *
import sys, os
from optimizers.optimizer import *

class PSO(optimizer):
    def __init__(self, info):
        self.info = info
        self.default_yield_params = None
        self.FirstStageOptimizer = None
        self.partialResult = None 
        self.SecondStageOptimizer = None
    
    def InitializeFirstStageOptimizer(self, default_yield_params):
        pass

    def FirstStageRun(self):
        pass

    def FirstStageOutputResult(self):
        pass

    def InitializeSecondStageOptimizer(self, optimized_yielding_params):
        pass

    def SecondStageRun(self):
        pass

    def SecondStageOutputResult(self):
        pass
