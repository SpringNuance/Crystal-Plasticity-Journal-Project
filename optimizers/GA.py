from modules.stoploss import *
from modules.helper import *
from modules.preprocessing import *
import pygad
import time

from math import *
from optimizers.optimizer import *



class GA(optimizer):

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
        self.num_generations=30
        self.num_parents_mating=120
        self.sol_per_pop=1200
        self.allow_duplicate_genes=False
        self.parent_selection_type="tournament"
        self.crossover_type="uniform"
        self.mutation_type="random"
        self.mutation_num_genes=1
        self.keep_elitism=120
        self.stop_criteria="saturate_8"

    ##########################
    # OPTIMIZATION FUNCTIONS #
    ##########################

    def initializeOptimizer(self, default_params, optimize_params, optimize_type):
        loadings = self.info["loadings"]
        param_info_GA_discrete = self.info["param_info_GA_discrete"] 
        param_info_GA_continuous = self.info["param_info_GA_continuous"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        exp_curves = self.info["exp_curves"] 
        regressors = self.info["regressors"]
        scalers = self.info["scalers"]
        weightsYielding = self.info["weightsYielding"]
        weightsHardening = self.info["weightsHardening"]
        weightsLoading = self.info["weightsLoading"]

        self.default_params = default_params
        self.optimize_params = optimize_params
        self.optimize_type = optimize_type 

        GA_bounds = []
        for param in self.optimize_params:
            if searchingSpace == "discrete":
                GA_bounds.append(param_info_GA_discrete[param])
            elif searchingSpace == "continuous":
                GA_bounds.append(param_info_GA_continuous[param])
        num_genes = len(GA_bounds)

        def fitnessGA(solution, solution_idx):
            default_params_dict = dict(self.default_params)
            counter = 0
            for param in self.optimize_params:
                if searchingSpace == "discrete":
                    default_params_dict[param] = solution[counter]
                elif searchingSpace == "continuous":
                    default_params_dict[param] = round(solution[counter], roundContinuousDecimals)
                counter += 1 
            candidate_params = np.array(list(default_params_dict.values())) 

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
  
            fitnessScore = 1/fitness
            return fitnessScore
        
        ga_instance = pygad.GA(num_generations=self.num_generations, 
                            num_parents_mating=self.num_parents_mating, 
                            sol_per_pop=self.sol_per_pop, 
                            num_genes=num_genes,
                            fitness_func=fitnessGA,
                            gene_space=GA_bounds,
                            allow_duplicate_genes=self.allow_duplicate_genes,
                            parent_selection_type=self.parent_selection_type,
                            crossover_type=self.crossover_type,
                            mutation_type=self.mutation_type,
                            mutation_num_genes=self.mutation_num_genes,
                            keep_elitism=self.keep_elitism,
                            stop_criteria=self.stop_criteria,
                            #parallel_processing=["thread", 5],
                            )
        self.optimizer = ga_instance

    def run(self):
        self.optimizer.run()

    def outputResult(self):
        param_info = self.info["param_info"] 
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]

        solution, solution_fitness, solution_idx = self.optimizer.best_solution(self.optimizer.last_generation_fitness)
        solution_fitness = 1/solution_fitness

        solution_dict = dict(self.default_params)
        counter = 0
        for param in self.optimize_params:
            if searchingSpace == "discrete":
                solution_dict[param] = round_to_step(param_info[param]['low'], param_info[param]['step'], solution[counter], param_info[param]['round'])
            elif searchingSpace == "continuous":
                solution_dict[param] = round(solution[counter], roundContinuousDecimals)
            counter += 1
        solution_tuple = tuple(solution_dict.items())
 
        output = {"solution_dict": solution_dict, "solution_tuple": solution_tuple, "solution_fitness": solution_fitness}
        return output