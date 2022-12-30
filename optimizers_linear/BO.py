from modules.stoploss import *
from modules.helper import *
import bayes_opt
from math import *
from optimizers.optimizer import *

class BO(optimizer):
    
    ##################################
    # OPTIMIZER CLASS INITIALIZATION #
    ##################################

    def multiply(self, tupleRange, multiplier):
        return tuple(int(i * multiplier) for i in tupleRange)
        
    def __init__(self, info):
        self.info = info
        self.default_hardening_params = None
        self.FirstStageOptimizer = None
        self.optimized_yielding_params = None 
        self.SecondStageOptimizer = None
    
    ######################################
    # FIRST STAGE OPTIMIZATION FUNCTIONS #
    ######################################
    
    def surrogateYieldBOgenerate(self, default_hardening_params):
        CPLaw = self.info["CPLaw"]
        weightsYield = self.info["weightsYield"]
        numberOfParams = self.info["numberOfParams"] 
        param_info = self.info["param_info"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        exp_stress = self.info["exp_stress"] 
        exp_strain = self.info["exp_strain"]
        regressor = self.info["regressor"]
        wy1 = weightsYield["wy1"]
        wy2 = weightsYield["wy2"]

        # Initialize surrogate function
        if CPLaw == "PH":
            def surrogateYieldBO(tau0):
                params = {
                    'a': default_hardening_params["a"],
                    'h0': default_hardening_params["h0"],
                    'tau0': tau0 * (10 ** - param_info["tau0"]["round"]),
                    'tausat': default_hardening_params["tausat"]
                }
                # Rounding is required because BO only deals with continuous values.
                # Rounding help BO probe at discrete parameters with correct step size
                if searchingSpace == "discrete":
                    candidate_dict_round = round_discrete(params, param_info)
                elif searchingSpace == "continuous":
                    candidate_dict_round = round_continuous(params, roundContinuousDecimals)
                solution = np.array(list(candidate_dict_round.values()))
                predicted_sim_stress = regressor.predict(solution.reshape(1, numberOfParams)).reshape(-1)
                candidateScore = fitnessYieldingOneLoading(exp_stress, predicted_sim_stress, exp_strain, wy1, wy2)
                fitnessScore = 1/candidateScore
                return fitnessScore
        elif CPLaw == "DB":
            def surrogateYieldBO(p, q, tausol):
                params = {
                    'dipole': default_hardening_params['dipole'],
                    'islip': default_hardening_params['islip'],
                    'omega': default_hardening_params['omega'],
                    'p': p * (10 ** - param_info["p"]["round"]),
                    'q': q * (10 ** - param_info["q"]["round"]), 
                    'tausol': tausol * (10 ** - param_info["tausol"]["round"])
                }
                # Rounding is required because BO only deals with continuous values.
                # Rounding help BO probe at discrete parameters with correct step size
                if searchingSpace == "discrete":
                    candidate_dict_round = round_discrete(params, param_info)
                elif searchingSpace == "continuous":
                    candidate_dict_round = round_continuous(params, roundContinuousDecimals)
                solution = np.array(list(candidate_dict_round.values()))
                predicted_sim_stress = regressor.predict(solution.reshape(1, numberOfParams)).reshape(-1)
                candidateScore = fitnessYieldingOneLoading(exp_stress, predicted_sim_stress, exp_strain, wy1, wy2)
                fitnessScore = 1/candidateScore
                return fitnessScore
        return surrogateYieldBO

    def InitializeFirstStageOptimizer(self, default_hardening_params):
        CPLaw = self.info["CPLaw"]
        param_info = self.info["param_info"]
        param_info_no_step_tuple = self.info["param_info_no_step_tuple"]
        self.default_hardening_params = default_hardening_params
        
        if CPLaw == "PH":
            pbounds = {
                "tau0": self.multiply(param_info_no_step_tuple['tau0'], 10 ** param_info["tau0"]["round"])
            }
        elif CPLaw == "DB":
            pbounds = {
                "p": self.multiply(param_info_no_step_tuple['p'], 10 ** param_info["p"]["round"]), 
                "q": self.multiply(param_info_no_step_tuple['q'], 10 ** param_info["q"]["round"]), 
                "tausol": self.multiply(param_info_no_step_tuple['tausol'], 10 ** param_info["tausol"]["round"])
            }
      
        surrogateYieldBO = self.surrogateYieldBOgenerate(default_hardening_params)
        bo_instance = bayes_opt.BayesianOptimization(f = surrogateYieldBO,
                                        pbounds = pbounds, verbose = 2,
                                        random_state = 4)
            

        self.FirstStageOptimizer = bo_instance

    def FirstStageRun(self):
        # There are two ways of using BO: the sequential or automatic way. 
        # To use sequential way, comment out automatic way, from init_points = ... until after the loop
        # To use automatic way, comment out sequential way, from iterations = ... until after the loop
        # Sequential way  

        # Low kappa = 1 means more exploitation for UCB
        # High kappa = 10 means more exploration for UCB
        # Low xi = 0 means more exploitation for EI and POI
        # High xi = 0.1 means more exploration for EI and POI
        '''
        utility = bayes_opt.UtilityFunction(kind="ei", kappa=10, xi = 0.1)
        iterations = 200
        init_points = 200
        blockPrint()

        self.FirstStageOptimzer.maximize(
            init_points = init_points, 
            n_iter = 0)
        for i in range(iterations):
            next_point = self.FirstStageOptimizer.suggest(utility)
            target = surrogateYieldBO(**next_point)
            self.FirstStageOptimizer.register(params=next_point, target=target)
            for param in next_point:
                original = next_point[param] * 10 ** - param_info[param]["round"]
                next_point[param] = original
            next_point = round_params(next_point, param_info)
            # print("#{} Result: {}; f(x) = {}.".format(i, next_point, target))
        enablePrint()
        '''
        # Automatic way
        init_points = 100
        iterations = 100
        blockPrint()
        for i in range(1):
            self.FirstStageOptimizer.maximize(
                init_points = init_points, 
                n_iter = iterations,    
                # What follows are GP regressor parameters
                acq="ucb", kappa=1, alpha=1)
        enablePrint()
        self.FirstStageOptimizer.set_gp_params(normalize_y=True)

    def FirstStageOutputResult(self):
        param_info = self.info["param_info"]
        CPLaw = self.info["CPLaw"]
        # Returning the details of the best solution in a dictionary.
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        solution_dict_original = self.FirstStageOptimizer.max["params"]
        solution_fitness = self.FirstStageOptimizer.max["target"]
        fitness = 1/solution_fitness
        if CPLaw == "PH":
            solution_dict = {
                'a': self.default_hardening_params['a'],
                'h0': self.default_hardening_params['h0'],
                'tau0': solution_dict_original['tau0'] * (10 ** - param_info["tau0"]["round"]),
                'tausat': self.default_hardening_params['tausat']
            }
        elif CPLaw == "DB":
            solution_dict = {
                'dipole': self.default_hardening_params['dipole'],
                'islip': self.default_hardening_params['islip'],
                'omega': self.default_hardening_params['omega'],
                'p': solution_dict_original["p"] * (10 ** - param_info["p"]["round"]),
                'q': solution_dict_original["q"] * (10 ** - param_info["q"]["round"]), 
                'tausol': solution_dict_original["tausol"] * (10 ** - param_info["tausol"]["round"])
            }
        if searchingSpace == "discrete":
            solution_dict = round_discrete(solution_dict, param_info)
        elif searchingSpace == "continuous":
            solution_dict = round_continuous(solution_dict, roundContinuousDecimals)
        solution_list = list(solution_dict.values())
        values = (solution_list, solution_dict, solution_fitness, fitness)
        keys = ("solution_list", "solution_dict", "solution_fitness", "fitness")
        output = dict(zip(keys, values))
        return output

    #######################################
    # SECOND STAGE OPTIMIZATION FUNCTIONS #
    #######################################

    def surrogateHardeningBOgenerate(self, optimized_yielding_params):
        CPLaw = self.info["CPLaw"]
        numberOfParams = self.info["numberOfParams"] 
        param_info = self.info["param_info"]
        exp_stress = self.info["exp_stress"] 
        exp_strain = self.info["exp_strain"]
        regressor = self.info["regressor"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        weightsHardening = self.info["weightsHardening"]
        wh1 = weightsHardening["wh1"]
        wh2 = weightsHardening["wh2"]
        wh3 = weightsHardening["wh3"]
        wh4 = weightsHardening["wh4"]
        # Initialize surrogate function
        if CPLaw == "PH":
            def surrogateHardeningBO(a, h0, tausat):
                params = {
                    'a': a * (10 ** - param_info["a"]["round"]),
                    'h0': h0 * (10 ** - param_info["h0"]["round"]),
                    'tau0': optimized_yielding_params["tau0"],
                    'tausat': tausat * (10 ** - param_info["tausat"]["round"])
                }
                # Rounding is required because BO only deals with continuous values.
                # Rounding help BO probe at discrete parameters with correct step size
                if searchingSpace == "discrete":
                    candidate_dict_round = round_discrete(params, param_info)
                elif searchingSpace == "continuous":
                    candidate_dict_round = round_continuous(params, roundContinuousDecimals)
                solution = np.array(list(candidate_dict_round.values()))
                predicted_sim_stress = regressor.predict(solution.reshape(1, numberOfParams)).reshape(-1)
                candidateScore = fitnessHardeningOneLoading(exp_stress, predicted_sim_stress, exp_strain, wh1, wh2, wh3, wh4)
                fitnessScore = 1/candidateScore
                return fitnessScore
        elif CPLaw == "DB":
            def surrogateHardeningBO(dipole, islip, omega):
                params = {
                    'dipole': dipole * (10 ** - param_info["dipole"]["round"]),
                    'islip': islip * (10 ** - param_info["islip"]["round"]),
                    'omega': omega * (10 ** - param_info["omega"]["round"]),
                    'p': optimized_yielding_params["p"],
                    'q': optimized_yielding_params["q"],
                    'tausol': optimized_yielding_params["tausol"]
                }
                # Rounding is required because BO only deals with continuous values.
                # Rounding help BO probe at discrete parameters with correct step size
                if searchingSpace == "discrete":
                    candidate_dict_round = round_discrete(params, param_info)
                elif searchingSpace == "continuous":
                    candidate_dict_round = round_continuous(params, roundContinuousDecimals)
                solution = np.array(list(candidate_dict_round.values()))
                predicted_sim_stress = regressor.predict(solution.reshape(1, numberOfParams)).reshape(-1)
                candidateScore = fitnessHardeningOneLoading(exp_stress, predicted_sim_stress, exp_strain, wh1, wh2, wh3, wh4)
                fitnessScore = 1/candidateScore
                return fitnessScore
        return surrogateHardeningBO
            
    def InitializeSecondStageOptimizer(self, optimized_yielding_params):
        CPLaw = self.info["CPLaw"]
        param_info = self.info["param_info"] 
        param_info_no_step_tuple = self.info["param_info_no_step_tuple"] 
 

        self.optimized_yielding_params = optimized_yielding_params
        if CPLaw == "PH":
            pbounds = {
                "a": self.multiply(param_info_no_step_tuple['a'], 10 ** param_info["a"]["round"]), 
                "h0": self.multiply(param_info_no_step_tuple['h0'], 10 ** param_info["h0"]["round"]), 
                "tausat": self.multiply(param_info_no_step_tuple['tausat'], 10 ** param_info["tausat"]["round"])
            }
        elif CPLaw == "DB":
            pbounds = {
                "dipole": self.multiply(param_info_no_step_tuple['dipole'], 10 ** param_info["dipole"]["round"]), 
                "islip": self.multiply(param_info_no_step_tuple['islip'], 10 ** param_info["islip"]["round"]), 
                "omega": self.multiply(param_info_no_step_tuple['omega'], 10 ** param_info["omega"]["round"])
            }


        # There are two ways of using BO: the sequential or automatic way. 
        # To use sequential way, comment out automatic way, from init_points = ... until after the loop
        # To use automatic way, comment out sequential way, from iterations = ... until after the loop
        surrogateHardeningBO = self.surrogateHardeningBOgenerate(optimized_yielding_params)
        bo_instance = bayes_opt.BayesianOptimization(f = surrogateHardeningBO,
                                        pbounds = pbounds, verbose = 2,
                                        random_state = 4)
        
        self.SecondStageOptimizer = bo_instance

    def SecondStageRun(self):
        # There are two ways of using BO: the sequential or automatic way. 
        # To use sequential way, comment out automatic way, from init_points = ... until after the loop
        # To use automatic way, comment out sequential way, from iterations = ... until after the loop
        # Sequential way  

        # Low kappa = 1 means more exploitation for UCB
        # High kappa = 10 means more exploration for UCB
        # Low xi = 0 means more exploitation for EI and POI
        # High xi = 0.1 means more exploration for EI and POI
        '''
        utility = bayes_opt.UtilityFunction(kind="ei", kappa=10, xi = 0.1)
        iterations = 200
        init_points = 200
        blockPrint()
        self.FirstStageOptimizer.maximize(
            init_points = init_points, 
            n_iter = 0)
        for i in range(iterations):
            next_point = self.SecondStageOptimizer.suggest(utility)
            target = surrogateHardeningBO(**next_point)
            self.SecondStageOptimizer.register(params=next_point, target=target)
            for param in next_point:
                original = next_point[param] * 10 ** - param_info[param]["round"]
                next_point[param] = original
            next_point = round_params(next_point, param_info)
            # print("#{} Result: {}; f(x) = {}.".format(i, next_point, target))
        enablePrint()
        '''
        # Automatic way
        init_points = 100
        iterations = 100
        blockPrint()
        for i in range(1):
            self.SecondStageOptimizer.maximize(
                init_points = init_points, 
                n_iter = iterations,    
                # What follows are GP regressor parameters
                acq="ucb", kappa=1, alpha=1)
        enablePrint()
        self.SecondStageOptimizer.set_gp_params(normalize_y=True)

    def SecondStageOutputResult(self):
        param_info = self.info["param_info"]
        CPLaw = self.info["CPLaw"]
        searchingSpace = self.info["searchingSpace"]
        roundContinuousDecimals = self.info["roundContinuousDecimals"]
        # Returning the details of the best solution in a dictionary.
        solution_dict_original = self.SecondStageOptimizer.max["params"]
        solution_fitness = self.SecondStageOptimizer.max["target"]
        fitness = 1/solution_fitness
        if CPLaw == "PH":
            solution_dict = {
                'a': solution_dict_original['a'] * (10 ** - param_info["a"]["round"]),
                'h0': solution_dict_original['h0'] * (10 ** - param_info["h0"]["round"]),
                'tau0': self.optimized_yielding_params['tau0'],
                'tausat': solution_dict_original['tausat'] * (10 ** - param_info["tausat"]["round"])
            }
        elif CPLaw == "DB":
            solution_dict = {
                'dipole': solution_dict_original['dipole'] * (10 ** - param_info["dipole"]["round"]),
                'islip': solution_dict_original['islip'] * (10 ** - param_info["islip"]["round"]),
                'omega': solution_dict_original['omega'] * (10 ** - param_info["omega"]["round"]),
                'p': self.optimized_yielding_params["p"],
                'q': self.optimized_yielding_params["q"], 
                'tausol': self.optimized_yielding_params["tausol"]
            }
        if searchingSpace == "discrete":
            solution_dict = round_discrete(solution_dict, param_info)
        elif searchingSpace == "continuous":
            solution_dict = round_continuous(solution_dict, roundContinuousDecimals)
        solution = list(solution_dict.values())
        values = (solution, solution_dict, solution_fitness, fitness)
        keys = ("solution", "solution_dict", "solution_fitness", "fitness")
        output = dict(zip(keys, values))
        return output



        
