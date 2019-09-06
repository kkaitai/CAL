from Examples import HighVar, Trap, Windy, WindyTrap

ModelList = ['Q_Agent', 'DoubleQ_Agent', 'WeightedQ_Agent', 'WeightedPlusQ_Agent',\
             'SpeedyQ_Agent', 'AccurateQ_Agent', 'NewAccurateQ_Agent']

GridSize = [3,5,9]

if __name__=="__main__1":

    l_rate = [0.8,0.8,0.7]
    for gs,lr in zip(GridSize,l_rate):
        path = 'Results/Grid/'
        HighVar(ModelList,gs,Pro=7, path= path,l_rate=lr)
    for gs,lr in zip(GridSize,l_rate):
        path = 'Results/Trap/'
        Trap(ModelList, gs, Pro=7, path=path,l_rate = lr)
    for gs,lr in zip(GridSize,l_rate):
        path = 'Results/WindyDet/'
        Windy(ModelList,gs,Pro=7, path= path,l_rate = lr)
    for gs,lr in zip(GridSize,l_rate):
        path = 'Results/WindyTrap/'
        WindyTrap(ModelList,gs,Pro=7, path= path,l_rate = lr)

    for gs in GridSize:
        path = 'Results_Linear/Grid/'
        HighVar(ModelList,gs,Pro=7, path= path,l_rate=1)
    for gs in GridSize:
        path = 'Results_Linear/Trap/'
        Trap(ModelList, gs, Pro=7, path=path,l_rate = 1)
    for gs in GridSize:
        path = 'Results_Linear/WindyDet/'
        Windy(ModelList,gs,Pro=7, path= path,l_rate = 1)
    for gs in GridSize:
        path = 'Results_Linear/WindyTrap/'
        WindyTrap(ModelList,gs,Pro=7, path= path,l_rate = 1)

if __name__ == "__main__":

    path = '1000/Results/WindyTrap/'
    WindyTrap(ModelList, 3, Pro=7, path=path, l_rate=0.8)