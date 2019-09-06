import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 16})

ModelList = ['Q_Agent', 'DoubleQ_Agent', 'WeightedQ_Agent', 'WeightedPlusQ_Agent',\
             'SpeedyQ_Agent', 'AccurateQ_Agent', 'NewAccurateQ_Agent']
names = {'Q_Agent':'q', 'DoubleQ_Agent':'d', 'WeightedQ_Agent':'w','WeightedPlusQ_Agent':'wp',\
         'SpeedyQ_Agent':'s', 'AccurateQ_Agent':'a', 'NewAccurateQ_Agent':'na'}
drawnames = {'Q_Agent':'Q-learning', 'DoubleQ_Agent':'Double Q', 'WeightedQ_Agent':'Averaged Q','WeightedPlusQ_Agent':'Weighted Q',\
         'SpeedyQ_Agent':'Speedy Q', 'AccurateQ_Agent':'Accurate Q', 'NewAccurateQ_Agent':'Combo Accurate Q'}

def drawground(ModelList, path, GridSize, var):
    fin_path = path+'q'+str(GridSize)+'/'
    if var=='r':
        vname = 'rper_'
    else:
        vname = 'MQ01_'
    for i in ModelList:
        data = np.load(fin_path+vname+names[i]+str(GridSize)+'.npy')
        if i=='SpeedyQ_Agent':
            plt.plot(data,label=drawnames[i],color = 'saddlebrown')
        elif i=='NewAccurateQ_Agent':
            plt.plot(data, label=drawnames[i], color='hotpink')
        else:
            plt.plot(data, label=drawnames[i])


def drawground_Com(ModelList, path, GridSize, var):
    fin_path1 = path[0]+'q'+str(GridSize)+'/'
    fin_path2 = path[1] + 'q' + str(GridSize) + '/'
    if var=='r':
        for i in ModelList:
            data = np.load(fin_path1+'rper_'+names[i]+str(GridSize)+'.npy')
            plt.plot(data,label=drawnames[i])
            data = np.load(fin_path2+'rper_'+names[i]+str(GridSize)+'.npy')
            plt.plot(data,label=drawnames[i]+'_Linear')

    elif var=='M':
        for i in ModelList:
            data = np.load(fin_path1+'MQ01_'+names[i]+str(GridSize)+'.npy')
            plt.plot(data,label=drawnames[i])
            data = np.load(fin_path2+'MQ01_'+names[i]+str(GridSize)+'.npy')
            plt.plot(data,label=drawnames[i]+'_Linear')

def draw(Models, path, draw_ground,Frame):
    plt.figure(1,figsize=(20,10))
    plt.subplots_adjust(left=0.08,right=0.84)

    plt.subplot(231)
    plt.margins(x=0)
    draw_ground(Models, path, 3, 'r')
    plt.title(r'$3 \times 3$ Grid World', y=1.03)
    plt.ylabel(r'r per step')
    plt.ylim(Frame['ylim'][0])
    plt.yticks(Frame['yticks'][0])
    plt.gca().tick_params(axis='x', labelbottom=False)
    plt.xticks(np.arange(0, 2e4 + 1, 1e4))

    plt.subplot(234)
    plt.margins(x=0)
    draw_ground(Models, path, 3, 'M')
    plt.plot(np.ones(20000) * Frame['opvalues'][0], linestyle='--', color='k')
    plt.xlabel('Number of steps')
    plt.ylabel(r'$\max_{a} Q(s,a)$')
    plt.yticks(Frame['yticks'][1])
    plt.ylim(Frame['ylim'][1])
    plt.xticks(np.arange(0, 2e4 + 1, 1e4), ['0', r'$1 \times 10^4$', r'$2 \times 10^4$'])

    plt.subplot(232)
    plt.margins(x=0)
    draw_ground(Models, path, 5, 'r')
    plt.title(r'$5 \times 5$ Grid World', y=1.03)
    plt.ylim(Frame['ylim'][2])
    plt.yticks(Frame['yticks'][2])
    plt.xticks(np.arange(0, 5e4 + 1, 2.5e4))
    plt.gca().tick_params(axis='x', labelbottom=False)

    plt.subplot(235)
    plt.margins(x=0)
    draw_ground(Models, path, 5, 'M')
    plt.plot(np.ones(50000) * Frame['opvalues'][1], linestyle='--', color='k')
    plt.xlabel('Number of steps')
    plt.yticks(Frame['yticks'][3])
    plt.xticks(np.arange(0, 5e4 + 1, 2.5e4), ['0', r'$2.5 \times 10^4$', r'$5 \times 10^4$'])
    plt.ylim(Frame['ylim'][3])

    plt.subplot(233)
    plt.margins(x=0)
    draw_ground(Models, path, 9, 'r')
    plt.title(r'$9 \times 9$ Grid World', y=1.03)
    plt.ylim(Frame['ylim'][4])
    plt.yticks(Frame['yticks'][4])
    plt.xticks(np.arange(0, 1e5 + 1, 5e4))
    plt.gca().tick_params(axis='x', labelbottom=False)

    plt.subplot(236)
    plt.margins(x=0)
    draw_ground(Models, path, 9, 'M')
    plt.plot(np.ones(100000) * Frame['opvalues'][2], linestyle='--', color='k')
    plt.xlabel('Number of steps')
    plt.yticks(Frame['yticks'][5])
    plt.xticks(np.arange(0, 1e5 + 1, 5e4), ['0', r'$5 \times 10^4$', r'$1 \times 10^5$'])
    plt.ylim(Frame['ylim'][5])

    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.2)
    plt.show()

def HighVarFrame_Linear():
    F = {}
    F['yticks'] = [[-1,-0.5,0],np.arange(-10, 31, 10),[-1,-0.5,0],np.arange(-20, 41, 20),[-1,-0.5,0],np.arange(-40, 61, 20)]
    F['ylim'] = [(-1.2,0.3), (-10,38), (-1.2,0.3), (-20,43),(-1.2,0.3),(-20,68)]
    F['opvalues'] =[0.36,0.248,0.38]
    return F

def TrapFrame_Linear():
    F = {}
    F['yticks'] = [[-1,-0.5,0],np.arange(-10, 31, 10),[-1,-0.5,0],np.arange(-20, 41, 20),[-1,-0.5,0],np.arange(-40, 61, 20)]
    F['ylim'] = [(-1.2,0.3), (-10,35), (-1.2,0.3), (-20,44),(-1.2,0.3),(-20,75)]
    F['opvalues'] =[0.36,0.248,0.38]
    return F

if __name__=="__main__":
    frame = TrapFrame_Linear()
    #draw(ModelList, 'Results_Linear/Trap/', drawground, frame)
    draw(['AccurateQ_Agent'], ['Results/Trap/', 'Results_Linear/Trap/'], drawground_Com, frame)