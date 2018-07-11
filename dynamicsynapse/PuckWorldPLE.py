#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:56:56 2017

@author: chitianqilin
"""
import numpy as np
from ple.games.puckworld import PuckWorld
from ple import PLE
import DynamicSynapseArray2D as DSA
import time

def softMax(AList):
    AList=np.array(AList)
    posibility = AList/(np.sum(AList))
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
    StatArray=np.array([StateDict[key] for key in KeyList])
    StatArray[4]-=StatArray[0]
    StatArray[5]-=StatArray[1]
    StatArray[6]-=StatArray[0]
    StatArray[7]-=StatArray[1]
    return StatArray
# %%
if __name__ == "__main__":
    TimOfRecording=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
       
    new = True #False #
    Info =0
    if new:
        NumberOfSynapses = np.array([2, 6])
        Weighters = 0.2*(np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])-0.5) + 0.3 * np.ones([NumberOfSynapses[0], NumberOfSynapses[1]]) # 0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)
        WeighteAmountPerSynapse = 1
    #    WeighterInAxon = WeighteAmountPerSynapse* NumberOfSynapses - Weighters.sum()
        WeighterVarRates = np.zeros(NumberOfSynapses)
        CW = 100
        tauWV = 40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
        aWV = 100
        rWV = 5000
        scale=1
        damping =2

#        SimulationTimeInterval = 30
        WeightersCentral = np.ones(NumberOfSynapses)*0.45+ 0.04*np.mgrid[0:NumberOfSynapses[0],0:NumberOfSynapses[1]][1]#(NumberOfSynapses) #* np.random.rand(NumberOfNeuron,NumberOfSynapses) # 0.6 * np.random.rand(NumberOfNeuron,NumberOfSynapses) #np.array([4, 1, 1, 1, 1])#np.ones(NumberOfSynapses)*0.4 + 0.3 * np.random.rand(NumberOfSynapses) #np.ones(NumberOfSynapses)/2 + [0.8, 0.1,0.1, 0.1, 0]  #[0.2, 0.1, 0]

#        dt = 20
#        CW = 100
#        tauWV = 20
#        aWV = 100
#        rWV = 5000# 5000
#        scale = 1
#        damping =2

#        WeightersCentral = np.ones(NumberOfSynapses) * 0.5 + 0.2 * np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])   # np.ones(NumberOfSynapses)/2+[0.2, 0.1,0.0, 0.2, 0]  #[0.2, 0.1, 0]
        WeighterVarDamping = np.ones(NumberOfSynapses) * damping  # [10,2,2,2,2]       #[2,2,2]
    #    Parameters = [CW, tauWV, aWV , rWV, scale, WeightersCentral,WeighterVarDamping]
        WeightersCentralUpdateRate = 0.000012
        DampingUpdateRate = 0.0000002
        
        ADSA = DSA.DynamicSynapseArray(NumberOfSynapses=NumberOfSynapses, CW=CW, tauWV=tauWV, aWV=aWV, rWV=rWV, scale=scale,
                                       WeightersCentral=WeightersCentral, WeighterVarDamping=WeighterVarDamping, WeighteAmountPerSynapse=WeighteAmountPerSynapse,
                                       Weighters=Weighters, WeighterVarRates=WeighterVarRates, WeightersCentralUpdateRate=WeightersCentralUpdateRate, DampingUpdateRate = DampingUpdateRate)
    game = PuckWorld()
    p = PLE(game,fps=30,display_screen=True, force_fps=False, state_preprocessor=state_preprocessor)
    p.init()
    #myAgent = MyAgent(p.getActionSet())
    nb_frames = 300000
    rewardLast = 0.0
    ModulatorAmount=0
    dt = 33
    points = 0
    ADSA.InitRecording(nb_frames) 
    Tracet = np.arange(nb_frames)*dt   
    for f in range(nb_frames):

#    while(True):
        if p.game_over(): #check if the game is over
            p.reset_game()
        obs = p.getGameState()# p.getScreenRGB()
        print('obs')
        print(obs)
        ADSA.StateUpdate()
        Weights=(ADSA.Weighters[:, :]-0.5)*2
        print('Weights')
        print(Weights)
        output=np.dot(Weights, obs[2:])
        print('output')
        print(output)
        if np.all(np.abs(output)<1):
            actionIndex = -1
            actionIndex0 = -1
        else:
            actionIndex0=softMax(np.abs(output))#np.argmax(output)+1
        print(actionIndex0)
        if actionIndex0==0:
            if np.sign(actionIndex0)==1:
                actionIndex = 0
            else:
                actionIndex = 3
        elif actionIndex0==1:
            if np.sign(actionIndex0)==1:
                actionIndex = 1
            else:
                actionIndex = 2
        action = p.getActionSet()[actionIndex]#ulrdn
        #[115, 100, 119, 97, None]
            # myAgent.pickAction(reward, obs)
        reward = p.act(action)
        print('reward: %f'%(reward))
        #points += (reward - rewardLast)-0.0001*dt*points
        points = (10+reward)/10+(reward - rewardLast) if reward>-10 else reward - rewardLast
        print('points: %f'%(points))
        ModulatorAmount = points
        if ModulatorAmount<0:
            ModulatorAmount=0
            
        ADSA.StepSynapseDynamics(dt, ModulatorAmount) 
        rewardLast=reward
    NeuonNumber=1
    newSlice= [slice(None)]*3
    newSlice[1]=NeuonNumber
    Traces = Tracet, ADSA.Trace['Weighters'][newSlice], ADSA.Trace['WeighterVarRates'][newSlice], ADSA.Trace['WeighterInAxon'][newSlice], ADSA.Trace['WeightersCentral'][newSlice], ADSA.Trace['WeighterVarDamping'][newSlice], ADSA.Trace['EquivalentVolume'][newSlice]
    figure1,figure2,figure3,figure4,figure5, figure6, figure7,figure8,figure9,ax = DSA.plot(TimOfRecording, Traces, path='/media/archive2T/chitianqilin/SimulationResult/DynamicSynapse/Plots/', savePlots=True) #path=
        