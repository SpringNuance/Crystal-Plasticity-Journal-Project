from modules.stoploss import *
from modules.helper import *
from modules.preprocessing import *
import pygad
import time
import copy
from math import *
from optimizers.optimizer import *



class GA(optimizer):

    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def __init__(self, info, prepared_data, trained_models):
        self.info = info
        self.prepared_data = prepared_data
        self.trained_models = trained_models
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

    def initializeOptimizer(self, default_params, optimize_params, lossFunction, weightsLoadings, weightsConstitutive):
        loadings = self.info["loadings"]
        param_info_GA = self.info["param_info_GA"] 
        exp_curve = self.prepared_data["exp_curve"] 
        regressors = self.trained_models["regressors"]
        scalers = self.trained_models["scalers"]

        self.default_params = default_params # This is a dict
        self.optimize_params = optimize_params 

        GA_bounds = []
        for param in self.optimize_params:
            GA_bounds.append(param_info_GA[param])

        num_genes = len(GA_bounds)
        #print(GA_bounds)
        #print(num_genes)
        #time.sleep(30)

        def lossGA(solution, solution_idx):
            default_params_copy = copy.deepcopy(self.default_params)
            counter = 0
            for param in self.optimize_params:
                default_params_copy[param] = solution[counter]
                counter += 1 
            candidate_params = np.array(list(default_params_copy.values())) 

            sim_curve = {}
            sim_curve["interpolate"] = {}
            for loading in loadings:
                if loading.startswith("linear"):
                    scaledParams = scalers[loading].transform(candidate_params.reshape(1, -1))
                    predicted_interpolate_sim_stress = regressors[loading].predict(scaledParams).flatten() # changing [[1,2,3...]] into [1,2,3,..]
                    sim_curve["interpolate"][loading] = {}
                    sim_curve["interpolate"][loading]["stress"] = predicted_interpolate_sim_stress
            loss = lossFunction(exp_curve["interpolate"], sim_curve["interpolate"], loadings, weightsLoadings, weightsConstitutive)

            lossScore = 1/loss
            return lossScore
        
        ga_instance = pygad.GA(num_generations=self.num_generations, 
                            num_parents_mating=self.num_parents_mating, 
                            sol_per_pop=self.sol_per_pop, 
                            num_genes=num_genes,
                            fitness_func=lossGA,
                            gene_space=GA_bounds,
                            allow_duplicate_genes=self.allow_duplicate_genes,
                            parent_selection_type=self.parent_selection_type,
                            crossover_type=self.crossover_type,
                            mutation_type=self.mutation_type,
                            mutation_num_genes=self.mutation_num_genes,
                            stop_criteria=self.stop_criteria,
                            #parallel_processing=["thread", 5],
                            )
        self.optimizer = ga_instance

    def run(self):
        self.optimizer.run()

    def outputResult(self):
        param_info = self.info["param_info"] 

        solution, solution_loss, solution_idx = self.optimizer.best_solution(self.optimizer.last_generation_fitness)
        solution_loss = 1/solution_loss

        solution_dict = dict(self.default_params)
        counter = 0
        
        for param in self.optimize_params:
            solution_dict[param] = round_to_step(param_info[param]['low'], param_info[param]['step'], solution[counter], param_info[param]['round'])
            counter += 1
        solution_tuple = tuple(solution_dict.items())
 
        output = {"solution_dict": solution_dict, "solution_tuple": solution_tuple, "solution_loss": solution_loss}
        return output