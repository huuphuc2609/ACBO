import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pickle
from core.CostBayesianOptimizer import CostBayesianOptimizer
from ndfunction import *

path_log = "./logs/"

Runs = 30
NumIterations = 60
givenBudget = 10000
methods = ["ei", "eips", "ei-cool", "proposed", "ei-cool-d"] # Use 'ei-cool-d' if select CArBO

curDateTime = datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss.txt')


# MABmethod = "GaussianTS"
input_function = "test1d"
# input_function = "Branin2d"
# input_function = "Griewank2d"
# input_function = "Rastrigin3d"
# input_function = "Schubert2d"
# input_function = "Mccormick2d"
# input_function = "Ackley5d"
# input_function = "Alpine4d"
# input_function = "Hartman6d"
# input_function = "Michalewicz10d"
# input_function = "Eggholder"
# input_function = "Schwefel2d"
# input_function = "Levy2d"
# input_function = "Hartman3d"
# input_function = "ANN6d"

datasetNum = 1
case = "c0"

for med in methods:
    method = med
    # input_function = "test1d"
    if method == "GaussianTS":
        MABmethod = "GaussianTS"
    elif method == "GP-Hedge":
        MABmethod = "GP-Hedge"
    else:
        MABmethod = "BayesianUCB"

    if input_function == "test1d":
        # myfunc = test1d(case)
        myfunc = test1d_shift(case)
    if input_function == "Branin2d":
        myfunc = Branin(case)
    if input_function == "Hartman3d":
        myfunc = Hartman_3d(case)
    if input_function == "Hartman6d":
        myfunc = Hartman_6d_shift(case)
    if input_function == "Schubert2d":
        # myfunc = Schubert_Nd(2,case)
        myfunc = Schubert_Nd_shift(2, case)
    if input_function == "Eggholder":
        myfunc = Eggholder(case)
    if input_function == "Levy2d":
        myfunc = Levy_Nd(2, case)
    if input_function == "Schwefel2d":
        myfunc = Schwefel_Nd(2, case)
    if input_function == "Griewank2d":
        # myfunc = Griewank_Nd(2, case)
        myfunc = Griewank_Nd_shift(2, case)
    if input_function == "Mccormick2d":
        myfunc = Mccormick2d(case)
    if input_function == "Ackley2d":
        myfunc = AckleyNd_shift(2, case)
    if input_function == "Ackley5d":
        myfunc = AckleyNd_shift(5, case)
    if input_function == "Rastrigin3d":
        myfunc = Rastrigin_Nd_shift(3, case)
    if input_function == "Alpine4d":
        myfunc = Alpine_Nd_shift(4, case)
    if input_function == "Michalewicz10d":
        myfunc = Michalewicz10d_shift(10, case)


    if myfunc.input_dim < 6:
        acq_opt_med = "DIRECT"  # < 6D
    else:
        acq_opt_med = "GA" # For 6D or above

    # Output optimizing process
    log_iter = ""
    log_turn = ""
    log_reward = ""
    log_everyTen = ""
    for trial in range(0, Runs):  # synthetic
        print("Trial:", trial)
        # Read from data
        if input_function == "test1d":
            dfX = pd.read_csv(r'data/test1d_3p_' + case + '/Test1dX' + str(trial) + '.csv')
        elif input_function == "Branin2d":
            dfX = pd.read_csv(r'data/branin_3p_'+case+'/BraninX' + str(trial) + '.csv')
            # dfX = pd.read_csv(r'data/branin_4p_c3/BraninX' + str(run) + '.csv')
        elif input_function == "Eggholder":
            dfX = pd.read_csv(r'data/eggholder_3p_'+case+'/EggholderX' + str(trial) + '.csv')
        elif input_function == "Levy2d":
            dfX = pd.read_csv(r'data/levy_3p_'+case+'/LevyNdX' + str(trial) + '.csv')
        elif input_function == "Hartman3d":
            dfX = pd.read_csv(r'data/hartman3d_10p_'+case+'/Hartman_3dX' + str(trial) + '.csv')
        elif input_function == "Hartman6d":
            # dfX = pd.read_csv(r'data/hartman6d_7p_'+case+'/Hartman_6dX' + str(trial) + '.csv')
            dfX = pd.read_csv(r'data/hartman6d_7p_' + case + '_init/Hartman_6dX' + str(trial) + '.csv')
        elif input_function == "Schubert2d":
            dfX = pd.read_csv(r'data/schubert2d_3p_'+case+'/Schubert2dX' + str(trial) + '.csv')
        elif input_function == "Schwefel2d":
            dfX = pd.read_csv(r'data/schwefel2d_3p_' + case + '/SchwefelNdX' + str(trial) + '.csv')
        elif input_function == "Griewank2d":
            dfX = pd.read_csv(r'data/griewank2d_3p_' + case + '/GriewankNdX' + str(trial) + '.csv')
        elif input_function == "Ackley2d":
            dfX = pd.read_csv(r'data/ackley2d_3p_' + case + '/AckleyNdX' + str(trial) + '.csv')
        elif input_function == "Ackley5d":
            # dfX = pd.read_csv(r'data/ackley5d_6p_' + case + '/AckleyNdX' + str(trial) + '.csv')
            dfX = pd.read_csv(r'data/ackley5d_6p_' + case + '_init/AckleyNdX' + str(trial) + '.csv')
        elif input_function == "Mccormick2d":
            dfX = pd.read_csv(r'data/mccormick2d_3p_' + case + '/Mccormick2dX' + str(trial) + '.csv')
        elif input_function == "Rastrigin3d":
            # dfX = pd.read_csv(r'data/rastrigin3d_4p_' + case + '/RastriginNdX' + str(trial) + '.csv')
            dfX = pd.read_csv(r'data/rastrigin3d_4p_' + case + '_init/RastriginNdX' + str(trial) + '.csv') # INIT DESIGN
        elif input_function == "Alpine4d":
            # dfX = pd.read_csv(r'data/alpine4d_5p_' + case + '/AlpineNdX' + str(trial) + '.csv')
            dfX = pd.read_csv(r'data/alpine4d_5p_' + case + '_init/AlpineNdX' + str(trial) + '.csv')
        elif input_function == "Michalewicz10d":
            # dfX = pd.read_csv(r'data/michalewicz10d_11p_' + case + '/Michalewicz10dX' + str(trial) + '.csv')
            dfX = pd.read_csv(r'data/michalewicz10d_11p_' + case + '_init/Michalewicz10dX' + str(trial) + '.csv')
        elif input_function == "ANN6d":
            dfX = pd.read_csv(r'data/ann_12p/mlpX' + str(trial) + '.csv')

        # dfX = pd.read_csv(r'data/griewank5d_3p_' + case + '/GriewankNdX' + str(trial) + '.csv')

        inX = dfX.values
        Xori = inX.tolist().copy()

        if input_function == "test1d":
            dfY = pd.read_csv(r'data/test1d_3p_' + case + '/Test1dY' + str(trial) + '.csv')
        elif input_function == "Branin2d":
            dfY = pd.read_csv(r'data/branin_3p_'+case+'/BraninY' + str(trial) + '.csv')
            # dfY = pd.read_csv(r'data/branin_4p_c3/BraninY' + str(run) + '.csv')
        elif input_function == "Eggholder":
            dfY = pd.read_csv(r'data/eggholder_3p_' + case + '/EggholderY' + str(trial) + '.csv')
        elif input_function == "Levy2d":
            dfY = pd.read_csv(r'data/levy_3p_' + case + '/LevyNdY' + str(trial) + '.csv')
        elif input_function == "Hartman3d":
            dfY = pd.read_csv(r'data/hartman3d_10p_'+case+'/Hartman_3dY' + str(trial) + '.csv')
        elif input_function == "Hartman6d":
            # dfY = pd.read_csv(r'data/hartman6d_7p_'+case+'/Hartman_6dY' + str(trial) + '.csv')
            dfY = pd.read_csv(r'data/hartman6d_7p_' + case + '_init/Hartman_6dY' + str(trial) + '.csv')
        elif input_function == "Schubert2d":
            dfY = pd.read_csv(r'data/schubert2d_3p_'+case+'/Schubert2dY' + str(trial) + '.csv')
        elif input_function == "Schwefel2d":
            dfY = pd.read_csv(r'data/schwefel2d_3p_' + case + '/SchwefelNdY' + str(trial) + '.csv')
        elif input_function == "Griewank2d":
            dfY = pd.read_csv(r'data/griewank2d_3p_' + case + '/GriewankNdY' + str(trial) + '.csv')
        elif input_function == "Ackley2d":
            dfY = pd.read_csv(r'data/ackley2d_3p_' + case + '/AckleyNdY' + str(trial) + '.csv')
        elif input_function == "Ackley5d":
            # dfY = pd.read_csv(r'data/ackley5d_6p_' + case + '/AckleyNdY' + str(trial) + '.csv')
            dfY = pd.read_csv(r'data/ackley5d_6p_' + case + '_init/AckleyNdY' + str(trial) + '.csv')
        elif input_function == "Mccormick2d":
            dfY = pd.read_csv(r'data/mccormick2d_3p_' + case + '/Mccormick2dY' + str(trial) + '.csv')
        elif input_function == "Rastrigin3d":
            # dfY = pd.read_csv(r'data/rastrigin3d_4p_' + case + '/RastriginNdY' + str(trial) + '.csv')
            dfY = pd.read_csv(r'data/rastrigin3d_4p_' + case + '_init/RastriginNdY' + str(trial) + '.csv')
        elif input_function == "Alpine4d":
            # dfY = pd.read_csv(r'data/alpine4d_5p_' + case + '/AlpineNdY' + str(trial) + '.csv')
            dfY = pd.read_csv(r'data/alpine4d_5p_' + case + '_init/AlpineNdY' + str(trial) + '.csv')
        elif input_function == "Michalewicz10d":
            # dfY = pd.read_csv(r'data/michalewicz10d_11p_' + case + '/Michalewicz10dY' + str(trial) + '.csv')
            dfY = pd.read_csv(r'data/michalewicz10d_11p_' + case + '_init/Michalewicz10dY' + str(trial) + '.csv')
        elif input_function == "ANN6d":
            dfY = pd.read_csv(r'data/ann_12p/mlpY' + str(trial) + '.csv')

        inY = dfY.values
        inY = inY.squeeze()
        inY = inY.tolist()
        print("inY:", inY)
        Yeval = [ins[0] for ins in inY]
        Ycost = [ins[1] for ins in inY]

        in_X = Xori
        print("Yeval:", Yeval)
        print("Ycost:", Ycost)

        BO = CostBayesianOptimizer(myfunc, initGuess=2, kerneltype="rbf", numIter=NumIterations, localOptAlgo=acq_opt_med,
                                   useLib=False, MABmethod=MABmethod, trial=trial)

        BOstart_time = time.time()
        bestBO, box, boy, ite, miny = BO.run(method=method, randomlyInit=False, budget=givenBudget, Xobs=in_X, Yobs_eval=Yeval, Yobs_cost=Ycost)
        BOstop_time = time.time()
        ite = ite - 1
        print("Run ", trial, ": CostBO x: ", bestBO, " y:", -1.0 * miny,
              " time: --- %s seconds ---" % (BOstop_time - BOstart_time))
        log_iter += BO.log_iter
        if med == "eips_pure" or med == "eips_pure_sep" or med == "proposed_pure" or med == "GaussianTS" or med == "GP-Hedge" or med == "ei_ucbps" or med == "ei-cool_ei":
            for turn in BO.turns:
                log_turn += str(turn) + ","
            log_turn+= '\n'
            log_reward += "Run " + str(trial) + '\n'
            for reward in BO.rewards:
                log_reward += str(reward) + ","
            log_reward += '\n'
            log_reward += "Arm0" + '\n'
            for idx in range(0,len(BO.turns)):
                if BO.turns[idx] == 0:
                    log_reward += str(BO.rewards[idx]) + ","
            log_reward += '\n'
            log_reward += "Arm1" + '\n'
            for idx in range(0,len(BO.turns)):
                if BO.turns[idx] == 1:
                    log_reward += str(BO.rewards[idx]) + ","
            log_reward += '\n'
        print("Len:", BO.GP.kernel.lengthScale," amp:", BO.GP.kernel.amp)

        # Save log every 10 runs / each case
        log_everyTen += BO.logEveryTen
        # if trial % 10 == 9:
        log_file_nameEvery10 = str(givenBudget) + " "
        if med == "eips_pure" or med == "eips_pure_sep" or med == "proposed_pure" or med == "GaussianTS" or med == "ei_ucbps" or med == "ei-cool_ei":
            log_file_nameEvery10 += str(
                NumIterations) + "cost" + input_function + "_" + method + "_" + case + "_" + MABmethod + curDateTime
        else:
            log_file_nameEvery10 += str(
                NumIterations) + "cost" + input_function + "_" + method + "_" + case + curDateTime
        log_file_nameEvery10 += "Every10_" + str(int(trial/10)) + ".txt"
        with open(path_log + log_file_nameEvery10, 'w') as output:
            # pickle.dump(log_everyTen, output, pickle.HIGHEST_PROTOCOL)
            output.write(log_everyTen)
        log_everyTen = ""
        log_file_name_turnEvery10 = str(NumIterations) + "cost" + input_function + "_" + method + "_" + case + "_" + MABmethod + "_turn"+ curDateTime + ".txt"
        log_file_name_turnEvery10 += "Every10_" + str(int(trial / 10)) + ".txt"
        with open(path_log + log_file_name_turnEvery10, 'w') as output:
            # pickle.dump(log_turn, output, pickle.HIGHEST_PROTOCOL)
            output.write(log_turn)
        # exit()
        BO.resetBO()

    strategy = ""
    strategy = "_fx_by_cx"

    log_file_name = str(givenBudget) + "_TruecostAddition_ " + acq_opt_med + "_"
    if med == "eips_pure" or med == "eips_pure_sep" or med == "proposed_pure" or med == "GaussianTS" or med == "ei_ucbps" or med == "ei-cool_ei" or med == "ei-cool-d" or med == "proposed-d":
        log_file_name += str(NumIterations) + "cost" + input_function + "_" + method + "_" + case + "_" + MABmethod + strategy + curDateTime +".txt"
    else:
        log_file_name += str(NumIterations) + "cost" + input_function + "_" + method + "_" + case + curDateTime + ".txt"

    with open(path_log + log_file_name, 'w') as output:
        # pickle.dump(log_iter, output, pickle.HIGHEST_PROTOCOL)
        output.write(log_iter)
    if med == "eips_pure" or med == "eips_pure_sep" or med == "proposed_pure" or med == "GaussianTS" or med == "GP-Hedge" or med == "ei_ucbps" or med == "ei-cool_ei" or med == "ei-cool-d" or med == "proposed-d":
        log_file_name_turn = str(NumIterations) + "cost" + input_function + "_" + method + "_" + case + "_" + MABmethod + "_turn"+ strategy + curDateTime + ".txt"
        with open(path_log + log_file_name_turn, 'w') as output:
            # pickle.dump(log_turn, output, pickle.HIGHEST_PROTOCOL)
            output.write(log_turn)
        log_file_name_reward = str(NumIterations) + "cost" + input_function + "_" + method + "_" + case + "_" + MABmethod + "_reward"+ strategy + curDateTime + ".txt"
        with open(path_log + log_file_name_reward, 'w') as output:
            # pickle.dump(log_reward, output, pickle.HIGHEST_PROTOCOL)
            output.write(log_reward)