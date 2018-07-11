#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:56:56 2017

@author: chitianqilin
"""
import numpy as np
from ple.games.puckworld import PuckWorld
from ple import PLE
import time
import LIFNeuron as LIFN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import os
import dill
from cycler import cycler
import shutil
import ButterLowpass as BLPF
ModelType = 'Bio'
if ModelType == 'Bio':
    import DynamicSynapseArray2D20180103 as DSA
elif ModelType == 'Eng':
    import DynamicSynapseArray2DRandomSin as DSA
    
def softMax(AList):
    AList=np.array(AList)
    posibility = AList.astype(float)/(np.sum(AList))
    roll = np.random.rand()
    accum = 0
    for i1 in range(len(posibility)):        
        if roll>accum and roll<accum+posibility[i1]:
            return i1
        accum += posibility[i1]
        
def state_preprocessor(StateDict):
    KeyList=["player_x",
              "player_y",
              "player_velocity_x",
            "player_velocity_y",
            "good_creep_x",
            "good_creep_y",
            "bad_creep_x",
            "bad_creep_y"]
    StatArray0=np.array([StateDict[key] for key in KeyList])
#    print(StateDict)
    StatArray0[4]-=StatArray0[0]
    StatArray0[5]-=StatArray0[1]
    StatArray0[6]-=StatArray0[0]
    StatArray0[7]-=StatArray0[1]
    StatArray=np.zeros((len(StatArray0)-2)*2)
#    print(StatArray0)
    for i in range(len(StatArray0)-2):
        if StatArray0[i+2]>0:
            StatArray[2*i+1]=StatArray0[i+2]
        else:
            StatArray[2*i]=-StatArray0[i+2]
    StatArray=StatArray[np.array([0,1,3,2,4,5,7,6,8,9,11,10])]
#    print(StatArray)
    return StatArray
# %%
if __name__ == "__main__":
    TimOfRecording=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    path='/media/archive2T/chitianqilin/SimulationResult/DynamicSynapse/PuckWorldPLESpike/DSPlots'+TimOfRecording+'/'
    new = True #False #

    LoadADSA = False
    Simulation = False
    fixedValue = False
    capturePath = path + 'Capture/'
    recording = True
    if recording:
        if not os.path.exists(path):
            os.makedirs(path) 
        if not os.path.exists(capturePath):
            os.makedirs(capturePath)  
    savePlots = False
    saveVideo = False
    
    linewidth= 1
    Info =0
    if new:
        NumberOfSynapses = np.array([2, 12])
        Weighters= np.ones(NumberOfSynapses)*0.2 + 0.1 * np.random.rand(NumberOfSynapses[0],NumberOfSynapses[1])#np.random.rand(NumberOfNeuron,NumberOfSynapses) #0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)  #
        WeighteAmountPerSynapse = 1
    #    WeighterInAxon = WeighteAmountPerSynapse* NumberOfSynapses - Weighters.sum()
        WeighterVarRates = np.zeros(NumberOfSynapses)
#        CW = 100
#        tauWV = 40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 130
#        rWV = 5000
#        scale=1
#        damping =2
#        CW = 100
#        tauWV =500#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#130   100
#        rWV = 7000 #7000
#        scale=1
#        damping =2
#        CW = 100
        
#        tauWV =0.1#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100#170#130   100
#        rWV = 7000*500*3000#7000
#        scale=1
#        damping =2*7000
#        CW = 100

        tauWV =0.5#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
        aWV = 100#170#130   100
        rWV = 7000*500#7000
        scale=1
        damping =2*7000
        CW = 100

        
        WeightersCentreUpdateCompensate = 0#0.4
        MaxDamping = 10*7000 #15*
#        SimulationTimeInterval = 30
        WeightersCentre = np.ones(NumberOfSynapses)*0.2+0.1 * np.random.rand(NumberOfSynapses[0],NumberOfSynapses[1])
#        0.04*np.mgrid[0:NumberOfSynapses[0],0:NumberOfSynapses[1]][1]#(NumberOfSynapses) #* np.random.rand(NumberOfNeuron,NumberOfSynapses) # 0.6 * np.random.rand(NumberOfNeuron,NumberOfSynapses) #np.array([4, 1, 1, 1, 1])#np.ones(NumberOfSynapses)*0.4 + 0.3 * np.random.rand(NumberOfSynapses) #np.ones(NumberOfSynapses)/2 + [0.8, 0.1,0.1, 0.1, 0]  #[0.2, 0.1, 0]

#        dt = 20
#        CW = 100
#        tauWV = 20
#        aWV = 100
#        rWV = 5000# 5000
#        scale = 1
#        damping =2

#        WeightersCentre = np.ones(NumberOfSynapses) * 0.5 + 0.2 * np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])   # np.ones(NumberOfSynapses)/2+[0.2, 0.1,0.0, 0.2, 0]  #[0.2, 0.1, 0]
        WeighterVarDamping = np.ones(NumberOfSynapses) * damping  # [10,2,2,2,2]       #[2,2,2]
    #    Parameters = [CW, tauWV, aWV , rWV, scale, WeightersCentre,WeighterVarDamping]
        WeightersCentreUpdateRate = 0.000030/10*4
        DampingUpdateRate = 0.0000002 /10000/2#2#/4
 
    if ModelType == 'Bio': 
        ADSA = DSA.DynamicSynapseArray(NumberOfSynapses=NumberOfSynapses, CW=CW, tauWV=tauWV, aWV=aWV, rWV=rWV, scale=scale,
                                       WeightersCentre=WeightersCentre, WeighterVarDamping=WeighterVarDamping, WeighteAmountPerSynapse=WeighteAmountPerSynapse,
                                       Weighters=Weighters, WeighterVarRates=WeighterVarRates, WeightersCentreUpdateRate=WeightersCentreUpdateRate, 
                                       DampingUpdateRate = DampingUpdateRate,WeightersCentreUpdateCompensate = WeightersCentreUpdateCompensate,MaxDamping = MaxDamping )
    elif ModelType == 'Eng':
        ADSA=DSA.DynamicSynapseArray(NumberOfSynapses , Period=20000, tInPeriod=None, PeriodVar=0.1,\
                 Amp=0.2, WeightersCentre = None, WeightersCentreUpdateRate = 0.000012, WeightersOscilateDecay=0.0000003/100) #Amp=0.2


    elif LoadADSA:
        with open('/media/archive2T/chitianqilin/SimulationResult/DynamicSynapse/PuckWorldPLESpike/DSPlots2017-10-02_22-21-09/ADSA2017-10-02_22-21-09.pkl') as ADSAPKL:
            ADSA = dill.load(ADSAPKL)  
        ADSAWeightersCentre = ADSA.WeightersCentre
#       \3
#    0-----1
#       \2
    indexRotation=[[0,1,2,3,4,5,6,7,8,9,10,11],  #u
                   [1,0,3,2,5,4,7,6,9,8,11,10], #d
                   [2,3,1,0,6,7,5,4,10,11,9,8], #l
                   [3,2,0,1,7,6,4,5,11,10,8,9]] #r
    game = PuckWorld()
    p = PLE(game,fps=30,display_screen=True, force_fps=True, state_preprocessor=state_preprocessor)
    p.init()
    #myAgent = MyAgent(p.getActionSet())
    nb_frames = 10*60*60*30
    
    StarRcordFrames=nb_frames-30*60*2
    
    rewardLast = 0.0
    ModulatorAmount=0
    dt = 100/3
    points = 0
    rewardAdaption=0
    rewardAdaptionRate=0.001
    recordingSample = 10
    if ModelType == 'Bio':
        ADSA.InitRecording(nb_frames/recordingSample) 
    elif ModelType == 'Eng':
        ADSA.InitRecording() 

#    Tracet = np.arange(nb_frames/recordingSample)*dt  
    Tracet = np.linspace(0, dt*nb_frames, nb_frames/recordingSample)   
    ALIFNArray=LIFN.LIFNeuronArray(NumberOfNuerons=4, VReset=-60, VFire=0, Vinit=-60, Capacity=1, LeakConduction=0.00, ISynapse=0)
    ALIFNArray.InitRecording(nb_frames/recordingSample)
    IPost = np.zeros(ALIFNArray.NumberOfNuerons)
    
    actionIndex = -1
    Trace={'t':Tracet,
           'actionIndex':np.zeros(nb_frames/recordingSample),
           'reward':np.zeros(nb_frames/recordingSample),
           'points':np.zeros(nb_frames/recordingSample),
           'rewardIncrease':np.zeros(nb_frames/recordingSample),
           'rewardAdaption':np.zeros(nb_frames/recordingSample)}
    for step in range(nb_frames):
        
#    while(True):
        if p.game_over(): #check if the game is over
            p.reset_game()
        obs = p.getGameState()# p.getScreenRGB()

        PBDistance=np.sqrt(np.sum(np.power(obs[8:],2)))
        if PBDistance<game.CREEP_BAD['radius_outer']: 
            FallinDistance = game.CREEP_BAD['radius_outer']-PBDistance
            xn=obs[9]/ PBDistance*FallinDistance
            xp=obs[8]/ PBDistance*FallinDistance
            yn=obs[11]/ PBDistance*FallinDistance
            yp=obs[10]/ PBDistance*FallinDistance            
            obs[8:]=np.array([xn,xp,yn,yp])

        if not fixedValue:
            if ModelType == 'Bio':
                ADSA.StateUpdate() 
            
            Weights=ADSA.Weighters[0, :]
#        Weights=np.array([0,0,0.5,0,0,0,0,2,0,0,1,0,])
        if fixedValue:
            Weights=ADSAWeightersCentre[0,:]
        for i1 in range(len(indexRotation)):
            IPost[i1]=np.sum(obs[indexRotation[i1]]*Weights)/2
#            IPost[i1]=np.sum(obs[indexRotation[i1]][np.array([2,3,6,7,10,11])]*Weights)
#            IPost[i1]=np.sum(obs[indexRotation[i1]][np.array([2,7,10])]*Weights[0:3])*3
     
#        IPost=np.dot(Weights, obs)
        
#        if np.all(np.abs(output)<1):
#            actionIndex = -1
#            actionIndex0 = -1
#        else:
#            actionIndex0=softMax(np.abs(output))#np.argmax(output)+1
#        print(actionIndex0)
#        if actionIndex0==0:
#            if np.sign(actionIndex0)==1:
#                actionIndex = 0
#            else:
#                actionIndex = 3
#        elif actionIndex0==1:
#            if np.sign(actionIndex0)==1:
#                actionIndex = 1
#            else:
#                actionIndex = 2

#        if np.all(np.less(IPost,1)):
#            actionIndex = -1
#        else:
#            actionIndex=softMax(IPost) 
            
        ALIFNArray.step(dt, IPost*0.015)
        if np.all(ALIFNArray.Spiking==False):
            actionIndex = -1
        else:
            actionIndex=softMax(ALIFNArray.Spiking) 
#        actionIndex =2
        action = np.array(p.getActionSet())[np.array([2,0,3,1,4])][actionIndex]#dru[udlrn] [0,3,1,2,4]
        #[115, 100, 119, 97, None]
            # myAgent.pickAction(reward, obs)
       
        reward = p.act(action)
        #points += (reward - rewardLast)-0.0001*dt*points
#        points = (10+reward)/10+(reward - rewardLast) if reward>-10 else reward - rewardLast
#        points = (10+reward)/10 
#        rewardIncrease=np.exp(reward/10)-np.exp(rewardLast/10)

#        rewardIncrease=np.exp(reward*3)-np.exp(rewardLast*3)
##        points = np.exp(rewardIncrease+1) if rewardIncrease>0 else 0#10
#        rewardAdaption += (rewardIncrease-rewardAdaption)*dt*rewardAdaptionRate
#        points = 2./(1+np.exp(-(rewardIncrease-rewardAdaption)*10))-1 if rewardIncrease-rewardAdaption>0 else 0

#
#        print('before reward: %f'%(reward))
#        print('before rewardAdaption: %f'%(rewardAdaption))   
        rewardAdaptionRate2 = 1 if reward > rewardAdaption else 0.5
        rewardAdaption += (reward-rewardAdaption)*dt*rewardAdaptionRate *rewardAdaptionRate2    
        rewardAfterAdapt = reward-rewardAdaption
#        print('After reward: %f'%(reward))
#        print('After rewardAdaption: %f'%(rewardAdaption))
        rewardIncrease=rewardAfterAdapt
#        points = 2./(1+np.exp(-rewardAfterAdapt*5))-1 if rewardAfterAdapt>0 else 0
        points = 1./(1+np.exp(-rewardAfterAdapt*10-5)) if rewardAfterAdapt>0 else 0

#        points = 2./(1+np.exp(-rewardAfterAdapt*5))-1
#        rewardAfterAdapt = reward+3
##        print('After reward: %f'%(reward))
##        print('After rewardAdaption: %f'%(rewardAdaption))
#        rewardIncrease=rewardAfterAdapt
#        points = 2./(1+np.exp(-rewardAfterAdapt))-1 if rewardAfterAdapt>0 else 0

#        points = rewardIncrease-rewardAdaption if rewardIncrease-rewardAdaption>0 else 0

        ModulatorAmount = points*0.2
        if ModulatorAmount<0 or nb_frames<150:
            ModulatorAmount=0
        if not fixedValue:            
            ADSA.StepSynapseDynamics(dt, ModulatorAmount, 'Dynamic0') 

        rewardLast=reward
#        print('points: %f'%(points))
        if step%300==0:
            print('step = %d'%(step))
            print('obs')
            print(obs)
            print('Weights')
            print(Weights)
            print('IPost')
            print(IPost)
            print('actionIndex')
            print(actionIndex)
            print('reward: %f'%(reward))
            print('rewardAfterAdapt: %f'%(rewardAfterAdapt))
            print('points: %f'%(points))
        if recording and step%recordingSample==0:
           Trace['actionIndex'][step/recordingSample]=actionIndex
           Trace['reward'][step/recordingSample]=reward
           Trace['points'][step/recordingSample]=points
           Trace['rewardIncrease'][step/recordingSample]=rewardIncrease
           Trace['rewardAdaption'][step/recordingSample]=rewardAdaption
           ADSA.Recording()
           ALIFNArray.Record()
        if saveVideo and step>StarRcordFrames:
            p.saveScreen(capturePath+'frame%.9d.png'%(p.getFrameNumber()))

        #%%  
    if not os.path.exists(path):
        os.makedirs(path) 
    codepath=path+'src/'
    if not os.path.exists(codepath):
        os.makedirs(codepath) 
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".py"): 
            shutil.copy2(filename, codepath)
    NeuonNumber=0
    newSlice= [slice(None)]*3
    newSlice[1]=NeuonNumber
    Traces = Tracet, ADSA.Trace['Weighters'][newSlice], ADSA.Trace['WeighterVarRates'][newSlice], ADSA.Trace['WeighterInAxonConcentration'][newSlice], ADSA.Trace['WeightersCentre'][newSlice], ADSA.Trace['WeighterVarDamping'][newSlice], ADSA.Trace['EquivalentVolume'][newSlice]
#    figure1,figure2,figure3,figure4,figure5, figure6, figure7,figure8,figure9,ax = DSA.plot(TimOfRecording, Traces, path=path, savePlots=savePlots, StartTimeRate=1, linewidth= linewidth) #path=
    FigerDict,ax = DSA.plot(TimOfRecording, Traces, path=path, savePlots=savePlots, StartTimeRate=1, linewidth= linewidth) #path=
#%%
    ALIFNArray.Plot(TimOfRecording, path=path+'LIFNPlots'+TimOfRecording, savePlots=savePlots, linewidth= linewidth)    
    Tracet=Trace['t'].astype(float)/1000
    pointFigure=plt.figure()
    LinePoints, = plt.plot(Tracet, Trace['points'], linewidth= linewidth)
    plt.ylabel('points')
    plt.xlabel('Time (s)')
    plt.title('points')
#%%    
    RewardPostProcessFigure=plt.figure()
    LineRewad,=plt.plot(Tracet, Trace['reward'], linewidth= linewidth)
    LineRewadIncrease,=plt.plot(Tracet, Trace['rewardIncrease'], linewidth= linewidth)
    LineRewardAdaption,=plt.plot(Tracet, Trace['rewardAdaption'], color='r', linewidth= linewidth)
    RewardLPF = BLPF.butter_lowpass_filter(data=Trace['reward'], cutoff=0.001, fs=1/Tracet[1], order=5)
    LineFilteredReward,=plt.plot(Tracet, RewardLPF)

    plt.legend([LineRewad, LineFilteredReward, LineRewardAdaption, LineRewadIncrease ],['Rewad', 'Filtered Reward', 'Reward Adaption','Reward after Adaption'])
    plt.xlabel('Time (s)')
    plt.title('Rewarod Post Process')
    RewardFigure= plt.figure()
    plt.plot(Tracet, Trace['reward'])
    plt.ylabel('Reward')
    plt.xlabel('Time (s)')
    plt.title('Reward')
    if savePlots == True:
        pp = PdfPages(path+"EnviromentPlots"+TimOfRecording+'.pdf')
        pointFigure.savefig(pp, format='pdf')
        RewardFigure.savefig(pp, format='pdf')
        RewardPostProcessFigure.savefig(pp, format='pdf')
        pp.close()
        Figures = {'pointFigure':pointFigure, ' RewardFigure': RewardFigure, 'RewardPostProcessFigure':RewardPostProcessFigure}
#        with open(path+"EnviromentPlots"+TimOfRecording+'.pkl', 'wb') as pkl:
#            dill.dump(Figures, pkl)
    with open(path+"ADSA"+TimOfRecording+'.pkl', 'wb') as pkl:
        dill.dump(ADSA, pkl)
    try:    
        dill.dump_session(path+"ADSA_session"+TimOfRecording+'.pkl' )
#    dill.dump_session(path+"Session"+TimOfRecording+'.pkl')
    except:
        print('dump_session error')
    #%%
    if saveVideo:
        os.system('ffmpeg -start_number %d -f image2 -r 30 -i '%(StarRcordFrames)+capturePath+'frame%9d.png  -vcodec mpeg4 -y '+capturePath+'movie.mp4')