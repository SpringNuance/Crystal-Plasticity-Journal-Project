from modules.stoploss import *
from modules.helper import *
import pygad
import time
from sklearn.metrics import mean_squared_error
from math import *
from optimizers.optimizer import *

last_fitness = 0
keep_parents = 1

class GA(optimizer):

    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def __init__(self, info):
        self.info = info
        self.default_hardening_params = None
        self.FirstStageOptimizer = None
        self.optimized_yielding_params = None 
        self.SecondStageOptimizer = None
    
    ######################################
    # FIRST STAGE OPTIMIZATION FUNCTIONS #
    ######################################

    def InitializeFirstStageOptimizer(self, default_hardening_params):
        CPLaw = self.info["CPLaw"]
        weightsYield = self.info["weightsYield"]
        numberOfParams = self.info["numberOfParams"] 
        param_range_no_round = self.info["param_range_no_round"] 
        param_range_no_step_dict = self.info["param_range_no_step_dict"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        exp_stress = self.info["exp_stress"] 
        exp_strain = self.info["exp_strain"]
        regressor = self.info["regressor"]
        wy1 = weightsYield["wy1"]
        wy2 = weightsYield["wy2"]

        self.default_hardening_params = default_hardening_params

        if searchingSpace == "discrete":
            if CPLaw == "PH":
                gene_space = [ param_range_no_round['tau0']]
                numberOfYieldStressParams = 1
            elif CPLaw == "DB":
                gene_space = [ param_range_no_round['p'], param_range_no_round['q'],  param_range_no_round['tausol']]
                numberOfYieldStressParams = 3
            num_genes = numberOfYieldStressParams


            def fitnessYieldGA(solution, solution_idx):
                if CPLaw == "PH":
                    partialSolution = np.array([default_hardening_params['a'], default_hardening_params['h0'], solution[0], default_hardening_params['tausat']])
                elif CPLaw == "DB":
                    partialSolution = np.array([default_hardening_params['dipole'], default_hardening_params['islip'], default_hardening_params['omega'], solution[0], solution[1], solution[2]])
                predicted_sim_stress = regressor.predict(partialSolution.reshape((1, numberOfParams))).reshape(-1)
                chromosomefit = fitness_yield(exp_stress, predicted_sim_stress, exp_strain, wy1, wy2)
                fitnessScore = 1/chromosomefit
                return fitnessScore
            
        elif searchingSpace == "continuous":
            if CPLaw == "PH":
                gene_space = [param_range_no_step_dict['tau0']]
                numberOfYieldStressParams = 1
            elif CPLaw == "DB":
                gene_space = [param_range_no_step_dict['p'], param_range_no_step_dict['q'],  param_range_no_step_dict['tausol']]
                numberOfYieldStressParams = 3
            num_genes = numberOfYieldStressParams

            def fitnessYieldGA(solution, solution_idx):
                if CPLaw == "PH":
                    partialSolution = np.array([default_hardening_params['a'], default_hardening_params['h0'], round(solution[0], roundContinuousDecimals), default_hardening_params['tausat']])
                elif CPLaw == "DB":
                    partialSolution = np.array([default_hardening_params['dipole'], default_hardening_params['islip'], default_hardening_params['omega'], round(solution[0], roundContinuousDecimals), round(solution[1], roundContinuousDecimals), round(solution[2], roundContinuousDecimals)])
                predicted_sim_stress = regressor.predict(partialSolution.reshape((1, numberOfParams))).reshape(-1)
                chromosomefit = fitness_yield(exp_stress, predicted_sim_stress, exp_strain, wy1, wy2)
                fitnessScore = 1/chromosomefit
                return fitnessScore
        
        def on_generation(ga_instance):
            global last_fitness
            generation = ga_instance.generations_completed
            fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
            last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        ga_instance = pygad.GA(num_generations=100, 
                            num_parents_mating=500, 
                            sol_per_pop=1000, 
                            num_genes=num_genes,
                            fitness_func=fitnessYieldGA,
                            on_generation=on_generation,
                            gene_space=gene_space,
                            crossover_type="single_point",
                            mutation_type="random",
                            mutation_num_genes=1)
        self.FirstStageOptimizer = ga_instance

    def FirstStageRun(self):
        self.FirstStageOptimizer.run()

    def FirstStageOutputResult(self):
        ga_instance = self.FirstStageOptimizer
        default_hardening_params = self.default_hardening_params
        CPLaw = self.info["CPLaw"]
        param_range = self.info["param_range"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        # Returning the details of the best solution in a dictionary.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        best_solution_generation = ga_instance.best_solution_generation
        fitness = 1/solution_fitness
        if CPLaw == "PH":
            solution_dict = {
                'a': default_hardening_params['a'],
                'h0': default_hardening_params['h0'],
                'tau0': solution[0],
                'tausat': default_hardening_params['tausat']
            }
        elif CPLaw == "DB":
            solution_dict = {
                'dipole': default_hardening_params['dipole'],
                'islip': default_hardening_params['islip'],
                'omega': default_hardening_params['omega'],
                'p': solution[0],
                'q': solution[1], 
                'tausol': solution[2]
            }
        if searchingSpace == "discrete":
            solution_dict = round_discrete(solution_dict, param_range)
            solution_list = list(solution_dict.values())
        elif searchingSpace == "continuous":
            solution_dict = round_continuous(solution_dict, roundContinuousDecimals)
            solution_list = list(solution_dict.values())
        values = (solution_list, solution_dict, solution_fitness, solution_idx, best_solution_generation, fitness)
        keys = ("solution", "solution_dict", "solution_fitness", "solution_idx", "best_solution_generation", "fitness")
        output = dict(zip(keys, values))
        return output

    #######################################
    # SECOND STAGE OPTIMIZATION FUNCTIONS #
    #######################################

    def InitializeSecondStageOptimizer(self, optimized_yielding_params):
        CPLaw = self.info["CPLaw"]
        weightsHardening = self.info["weightsHardening"]
        numberOfParams = self.info["numberOfParams"] 
        param_range_no_round = self.info["param_range_no_round"] 
        param_range_no_step_dict = self.info["param_range_no_step_dict"]
        exp_stress = self.info["exp_stress"] 
        exp_strain = self.info["exp_strain"]
        searchingSpace = self.info["searchingSpace"]  
        regressor = self.info["regressor"] 
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        wh1 = weightsHardening["wh1"]
        wh2 = weightsHardening["wh2"]
        wh3 = weightsHardening["wh3"]
        wh4 = weightsHardening["wh4"]
        self.optimized_yielding_params = optimized_yielding_params

        if searchingSpace == "discrete":
            if CPLaw == "PH":
                gene_space = [param_range_no_round['a'], param_range_no_round['h0'], param_range_no_round['tausat']]
                numberOfHardeningParams = 3
            elif CPLaw == "DB":
                gene_space = [param_range_no_round['dipole'], param_range_no_round['islip'], param_range_no_round['omega']]
                numberOfHardeningParams = 3
            num_genes = numberOfHardeningParams
            
            def fitnessHardeningGA(solution, solution_idx):
                if CPLaw == "PH":
                    fullSolution = np.array([solution[0], solution[1], optimized_yielding_params['tau0'], solution[2]])
                elif CPLaw == "DB":
                    fullSolution = np.array([solution[0], solution[1], solution[2], optimized_yielding_params['p'], optimized_yielding_params['q'], optimized_yielding_params['tausol']])
                predicted_sim_stress = regressor.predict(fullSolution.reshape((1, numberOfParams))).reshape(-1)
                chromosomefit = fitness_hardening(exp_stress, predicted_sim_stress, exp_strain, wh1, wh2, wh3, wh4)
                fitnessScore = 1/chromosomefit
                return fitnessScore


        elif searchingSpace == "continuous":
            if CPLaw == "PH":
                gene_space = [param_range_no_step_dict['a'], param_range_no_step_dict['h0'], param_range_no_step_dict['tausat']]
                numberOfHardeningParams = 3
            elif CPLaw == "DB":
                gene_space = [param_range_no_step_dict['dipole'], param_range_no_step_dict['islip'], param_range_no_step_dict['omega']]
                numberOfHardeningParams = 3
            num_genes = numberOfHardeningParams
            
            def fitnessHardeningGA(solution, solution_idx):
                if CPLaw == "PH":
                    fullSolution = np.array([round(solution[0], roundContinuousDecimals), round(solution[1], roundContinuousDecimals), optimized_yielding_params['tau0'], round(solution[2], roundContinuousDecimals)])
                elif CPLaw == "DB":
                    fullSolution = np.array([round(solution[0], roundContinuousDecimals), round(solution[1], roundContinuousDecimals), round(solution[2], roundContinuousDecimals), optimized_yielding_params['p'], optimized_yielding_params['q'], optimized_yielding_params['tausol']])
                predicted_sim_stress = regressor.predict(fullSolution.reshape((1, numberOfParams))).reshape(-1)
                chromosomefit = fitness_hardening(exp_stress, predicted_sim_stress, exp_strain, wh1, wh2, wh3, wh4)
                fitnessScore = 1/chromosomefit
                return fitnessScore

        def on_generation(ga_instance):
            global last_fitness
            generation = ga_instance.generations_completed
            fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
            change = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness
            last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

        ga_instance = pygad.GA(num_generations=100, # Number of generations.
                            num_parents_mating=500, # Number of solutions to be selected as parents in the mating pool.
                            sol_per_pop=1000, # Number of solutions in the population.
                            num_genes=num_genes,
                            fitness_func=fitnessHardeningGA,
                            on_generation=on_generation,
                            gene_space=gene_space,
                            crossover_type="single_point",
                            mutation_type="random",
                            mutation_num_genes=1)
        
        self.SecondStageOptimizer = ga_instance
    
    def SecondStageRun(self):
        self.SecondStageOptimizer.run()

    def SecondStageOutputResult(self):
        CPLaw = self.info["CPLaw"]
        optimized_yielding_params = self.optimized_yielding_params
        ga_instance = self.SecondStageOptimizer
        param_range = self.info["param_range"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        # Returning the details of the best solution in a dictionary.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        best_solution_generation = ga_instance.best_solution_generation
        fitness = 1/solution_fitness

        if CPLaw == "PH":
            solution_dict = {
                'a': solution[0],
                'h0': solution[1],
                'tau0': optimized_yielding_params['tau0'],
                'tausat': solution[2]
            }
        elif CPLaw == "DB":
            solution_dict = {
                'dipole': solution[0],
                'islip': solution[1],
                'omega': solution[2],
                'p': optimized_yielding_params['p'],
                'q': optimized_yielding_params['q'], 
                'tausol': optimized_yielding_params['tausol'],
            }

        if searchingSpace == "discrete":
            solution_dict = round_discrete(solution_dict, param_range)
            solution_list = list(solution_dict.values())
        elif searchingSpace == "continuous":
            solution_dict = round_continuous(solution_dict, roundContinuousDecimals)
            solution_list = list(solution_dict.values())
        values = (solution_list, solution_dict, solution_fitness, solution_idx, best_solution_generation, fitness)
        keys = ("solution", "solution_dict", "solution_fitness", "solution_idx", "best_solution_generation", "fitness")
        output = dict(zip(keys, values))
        return output






