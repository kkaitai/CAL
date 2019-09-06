import numpy as np
from Models import Q_Agent, DoubleQ_Agent, WeightedQ_Agent, WeightedPlusQ_Agent, SpeedyQ_Agent, AccurateQ_Agent, NewAccurateQ_Agent
from Grids import Grid, TrapGrid, WindyGrid, WindyTrapGrid, Play
import multiprocessing as mp

Model = {'Q_Agent':Q_Agent, 'DoubleQ_Agent':DoubleQ_Agent, 'WeightedQ_Agent':WeightedQ_Agent,\
        'WeightedPlusQ_Agent':WeightedPlusQ_Agent, 'SpeedyQ_Agent':SpeedyQ_Agent,\
         'AccurateQ_Agent':AccurateQ_Agent, 'NewAccurateQ_Agent':NewAccurateQ_Agent}

names = {'Q_Agent':'q', 'DoubleQ_Agent':'d', 'WeightedQ_Agent':'w','WeightedPlusQ_Agent':'wp',\
         'SpeedyQ_Agent':'s', 'AccurateQ_Agent':'a', 'NewAccurateQ_Agent':'na'}

def GroundSetting(GridSize):
    if GridSize == 3:
        DF = 0.95
        N_step = 20000
        FinalReward = [5, 5]
    elif GridSize == 5:
        DF = 0.96
        N_step = 50000
        FinalReward = [10, 10]
    elif GridSize == 9:
        DF = 0.98
        N_step = 100000
        FinalReward = [20, 20]
    return DF, N_step, FinalReward

def Ground(ModelList,GridSize,Pro, path, Trans ,TrapReward,NonReward, l_rate):
    DF, N_step, FinalReward = GroundSetting(GridSize)
    N_ex=50
    p = mp.Pool(processes=Pro)

    for i in ModelList:
        if i == 'WeightedQ_Agent':
            Agent = Model[i](GridSize,N_ex,l_rate, np.array([[0.5, 0.5],[0.5,0.5]]))
        elif i == 'WeightedPlusQ_Agent':
            Agent =  Model[i](GridSize,N_ex,l_rate,10)
        else:
            Agent = Model[i](GridSize,N_ex,l_rate)
        para = (GridSize, Trans, Agent, N_ex, N_step, DF, FinalReward, TrapReward, NonReward)
        results = []
        for j in range(20):
            results.append(p.apply_async(Play, para))
        rper = np.mean([res.get()[0] for res in results], axis=0)
        MQ0 = np.mean([res.get()[1] for res in results], axis=0)
        if path!=None:
            np.save(path + 'q'+str(GridSize)+'/MQ01_'+names[i]+str(GridSize)+'.npy',MQ0)
            np.save(path + 'q' + str(GridSize) + '/rper_' + names[i] + str(GridSize) + '.npy', rper)

def HighVar(ModelList,GridSize,Pro, path, l_rate):
    TrapReward = None
    NonReward = [-12, 10]
    Trans = Grid(GridSize)
    Ground(ModelList, GridSize, Pro, path, Trans, TrapReward, NonReward,l_rate)

def Trap(ModelList,GridSize,Pro, path,l_rate):
    TrapReward = 0
    NonReward = [-12, 10]
    Trans = TrapGrid(GridSize)
    Ground(ModelList, GridSize, Pro, path, Trans, TrapReward, NonReward,l_rate)

def Windy(ModelList,GridSize,Pro, path,l_rate):
    TrapReward = None
    NonReward = [-1, -1]
    Trans = WindyGrid(GridSize)
    Ground(ModelList, GridSize, Pro, path, Trans, TrapReward, NonReward,l_rate)

def WindyTrap(ModelList,GridSize,Pro, path,l_rate):
    TrapReward = 0
    NonReward = [-12, 10]
    Trans = WindyTrapGrid(GridSize)
    Ground(ModelList, GridSize, Pro, path, Trans, TrapReward, NonReward,l_rate)
