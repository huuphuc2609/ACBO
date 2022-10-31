import copy
import numpy as np
import pandas as pd


#Mac get rid of error of matplotlib
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib

from copy import deepcopy
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets

from ndfunction import *

matplotlib.use('TkAgg')
# np.random.seed(2609)

# myfunction = test1d("c0")
# myfunction = Branin("c2")
# myfunction = Eggholder("c3")
# myfunction = Levy_Nd(2,"c3")
# myfunction = Schwefel_Nd(2,"c3")
# myfunction = Griewank_Nd(2,"c1")
# myfunction = Hartman_3d("c2")
# myfunction = Schubert_Nd(2,"c3")
# myfunction = Mccormick2d("c3")
# myfunction = ANN()

# myfunction = test1d_shift("c1")
myfunction = Griewank_Nd_shift(2,"c8")
# myfunction = Schubert_Nd_shift(2,"c1")
myfunction = AckleyNd_shift(5,"c8")
# myfunction = Alpine_Nd_shift(4,"c8")
# myfunction = Hartman_6d_shift("c8")
myfunction = Michalewicz10d_shift(10,"c8")
# myfunction = Rastrigin_Nd_shift(3,"c8")

def objective(X, run):
    return myfunction.func(X, run)
    # return myfunction.robotSim(X)
    # return myfunction.heatSim(X)[0]

# for run in range(123,124): # Normal
# for run in range(0, 10): # Real00
# for run in range(1, 2): # 7, 13, 19, 25
# for run in [1, 7, 13, 19, 25]:
for run in range(0, 210): # Real03
    ### Init ###
    initX = []
    # 30 25 - Apline5d
    # 50 45 - hartman6d, 3d
    # 100 90 - Ackley10d
    # numberOfInit = myfunction.input_dim*10  # ANN
    numberOfInit = myfunction.input_dim + 1

    # numberOfInit = 3
    # numberOfInit = 12 # ANN
    initY = []
    checkY = []
    for i in range(0, numberOfInit):
        tmp = myfunction.randUniformInBounds()
        while tmp in initX:
            tmp = myfunction.randUniformInBounds()
        # tmp = [int(ins) for ins in tmp]
        tmpY = objective(tmp, run)
        tmpCheck = tmpY[0]
        while tmpCheck in checkY:
            tmp = myfunction.randUniformInBounds()
            tmpY = objective(tmp, run)
            tmpCheck = tmpY[0]

        initX.append(tmp)
        initY.append(tmpY)
        checkY.append(tmpCheck)
    # initY = [objective(i, run) for i in initX]
    ### Write to CSV - Using Pandas ###
    dataX = {}
    colsX = []
    keyStr = "x"
    for i in range(0, len(myfunction.bounds)):
        key = keyStr + str(i + 1)
        colsX.append(key)
        tmpCol = {}
        tmpCol[key] = [tmpX[i] for tmpX in initX]
        dataX.update(tmpCol)

    dtInitY = {}

    colsY = []
    colsY.append('y')
    colsY.append('cost')
    tmpCol = {}
    print("initY:",initY)
    tmpCol['y'] = [tmpY[0] for tmpY in initY]  # Evaluation
    tmpCol['cost'] = [tmpY[1] for tmpY in initY]  # Cost
    print("tmpCol:",tmpCol)
    dtInitY.update(tmpCol)

    dfX = pd.DataFrame(dataX, columns=colsX)
    export_csv = dfX.to_csv(myfunction.name + "X" + str(run) + ".csv", index=None, header=True)
    dfY = pd.DataFrame(dtInitY, columns=colsY)
    export_csv = dfY.to_csv(myfunction.name + "Y" + str(run) + ".csv", index=None, header=True)


