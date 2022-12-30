from modules.stoploss import *
from modules.helper import *
import bayes_opt
import time
from sklearn.metrics import mean_squared_error
from math import *
import sys, os
from optimizers.optimizer import *
from modules.stoploss import *
from modules.helper import *
from modules.preprocessing import *
import pygad
import time
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

from math import *
from optimizers.optimizer import *

class PSO(optimizer):

    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def __init__(self, info):
        self.info = info
        self.default_params = None
        self.optimize_params = None
        self.optimize_type = None

        #############################
        # Optimizer hyperparameters #
        #############################

        # options : dict with keys :code:`{'c1', 'c2', 'w'}`
        #     a dictionary containing the parameters for the specific optimization technique.
        #     c1: cognitive parameter
        #     c2: social parameter
        #     w: inertia parameter
        self.options = {'c1': 2.05, 'c2': 2.05, 'w': 0.72948}
        self.n_particles = 200
        self.iterations = 100
        self.velocity_clamp = (0.00001, 100)
        self.verbose = False
    
    ##########################
    # OPTIMIZATION FUNCTIONS #
    ##########################

    def initializeOptimizer(self, default_params, optimize_params, optimize_type):
        param_info = self.info["param_info"]
        param_infos_PSO_low = self.info["param_infos_PSO_low"]
        param_infos_PSO_high = self.info["param_infos_PSO_high"]

        self.default_params = default_params
        self.optimize_params = optimize_params
        self.optimize_type = optimize_type 

        PSO_bounds_low = []
        PSO_bounds_high = []
        for param in self.optimize_params:
            PSO_bounds_low.append(param_infos_PSO_low[param]["low"] *  (10 ** param_info[param]["round"])) 
            PSO_bounds_high.append(param_infos_PSO_high[param]["high"] *  (10 ** param_info[param]["round"]))
        num_parameters = len(PSO_bounds_low)

        PSO_bounds = (np.array(PSO_bounds_low), np.array(PSO_bounds_high))

        # A bound should be of type tuple with length 2.
        # It should contain two numpy.ndarrays so that we have a (min_bound, max_bound)
        # Obviously, all values in the max_bound should always be greater than the min_bound. Their shapes should match the dimensions of the swarm.
        
        # https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/single/global_best.py
        pso_instance =  ps.single.GlobalBestPSO(
            n_particles=self.n_particles, 
            dimensions=num_parameters, 
            options=self.options,
            velocity_clamp=self.velocity_clamp,
            bounds=PSO_bounds)

        self.optimizer = pso_instance

    def run(self):
        loadings = self.info["loadings"]
        searchingSpace = self.info["searchingSpace"]
        regressors = self.info["regressors"]
        scalers = self.info["scalers"]        
        exp_curves = self.info["exp_curves"] 
        weightsYielding = self.info["weightsYielding"]
        weightsHardening = self.info["weightsHardening"]
        weightsLoading = self.info["weightsLoading"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        param_info = self.info["param_info"]
        
        # solution is np array of shape (n_particles, number of params)
        # The fitness solution is returned with shape (number of params, ) (vector)
        def fitnessPSO(solutions):

            candidate_params_array = []
            num_particles = solutions.shape[0]
            for solution_index in range(0, num_particles):
                default_params_dict = dict(self.default_params)
                solution = solutions[solution_index, :]
                counter = 0
                for param in self.optimize_params:
                    scaledDownSolution = solution[counter] * (10 ** - param_info[param]["round"])
                    if searchingSpace == "discrete":
                        default_params_dict[param] = round_to_step(param_info[param]['low'], param_info[param]['step'], scaledDownSolution, param_info[param]['round'])
                    elif searchingSpace == "continuous":
                        default_params_dict[param] = round(scaledDownSolution, roundContinuousDecimals)
                    counter += 1 
                candidate_params_array.append(np.array(list(default_params_dict.values())))

            fitnessScores = [] # Dimension (num_particles, )
            candidate_params_array = np.array(candidate_params_array)
            #print(candidate_params_array.shape)
            for solution_index in range(0, num_particles):
                candidate_params = candidate_params_array[solution_index, :]
                if self.optimize_type == "yielding":
                    scaledParams = scalers["linear_uniaxial_RD"].transform(candidate_params.reshape(1, -1))
                    predicted_sim_stress = regressors["linear_uniaxial_RD"].predict(scaledParams).flatten() # changing [[1,2,3...]] into [1,2,3,..]
                    fitness = fitnessYieldingLinear(exp_curves["interpolate"]["linear_uniaxial_RD"]["stress"], predicted_sim_stress, exp_curves["interpolate"]["linear_uniaxial_RD"]["strain"], weightsYielding)
                elif self.optimize_type == "hardening":
                    fitness = 0
                    for loading in loadings:
                        scaledParams = scalers[loading].transform(candidate_params.reshape(1, -1))
                        predicted_sim_stress = regressors[loading].predict(scaledParams).flatten() # changing [[1,2,3...]] into [1,2,3,..]
                        if loading == "linear_uniaxial_RD":
                            fitness += weightsLoading[loading] * fitnessHardeningLinear(exp_curves["interpolate"][loading]["stress"], predicted_sim_stress,  exp_curves["interpolate"][loading]["strain"], weightsHardening)
                        else:
                            fitness += weightsLoading[loading] * fitnessHardeningNonlinear(exp_curves["interpolate"][loading]["stress"], predicted_sim_stress,  exp_curves["interpolate"][loading]["strain"], weightsHardening)

                fitnessScores.append(fitness)
            

            fitnessScores = np.array(fitnessScores)
            #print(fitnessScores)
            #time.sleep(30)
            return fitnessScores

        self.solution_fitness, self.solution = self.optimizer.optimize(fitnessPSO, iters=self.iterations, verbose=self.verbose)

    def outputResult(self):
        param_info = self.info["param_info"] 
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]

        solution, solution_fitness = self.solution, self.solution_fitness

        solution_dict = dict(self.default_params)
        
        counter = 0
        for param in self.optimize_params:
            scaledDownSolution = solution[counter] * (10 ** - param_info[param]["round"])
            if searchingSpace == "discrete":
                solution_dict[param] = round_to_step(param_info[param]['low'], param_info[param]['step'], scaledDownSolution, param_info[param]['round'])
            elif searchingSpace == "continuous":
                solution_dict[param] = round(scaledDownSolution, roundContinuousDecimals)
            counter += 1
        solution_tuple = tuple(solution_dict.items())
 
        output = {"solution_dict": solution_dict, "solution_tuple": solution_tuple, "solution_fitness": solution_fitness}
        return output
