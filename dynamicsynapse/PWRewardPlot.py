#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:49:29 2017

@author: chitianqilin
"""
import matplotlib.pyplot as plt
import dill
import ButterLowpass as BLPF
# %%
if __name__ == "__main__":
    RewardSessionFile="/media/archive2T/chitianqilin/SimulationResult/DynamicSynapse/PuckWorldPLESpike/DSPlots2017-10-02_22-21-09/Reward2017-10-02_22-21-09_2.pkl"

    with open (RewardSessionFile, 'rb') as pkl:
        RewardDataDict=dill.load(pkl)
    

# %%     RewardDataDict['figure']

    y1 = BLPF.butter_lowpass_filter(data=RewardDataDict['reward'], cutoff=0.001, fs=1/RewardDataDict['t'][1], order=5)
    linewidth=1
    Tracet=RewardDataDict['t']
    RewardPostProcessFigure=plt.figure()
    LineRewad,=plt.plot(Tracet, RewardDataDict['reward'], linewidth= linewidth)
    
    LineRewadafterAdaption,=plt.plot(Tracet, RewardDataDict['RewardAfterAdaption'], linewidth= linewidth)
    LineRewardAdaption,=plt.plot(Tracet, RewardDataDict['RewardAdaption'], color='r', linewidth= linewidth)
    LineFilteredReward,=plt.plot(RewardDataDict['t'],y1)
    plt.legend([LineRewad, LineFilteredReward, LineRewardAdaption, LineRewadafterAdaption ],['Rewad', 'Filtered Reward','Reward Adaption','Reward after Adaption'])
    plt.xlabel('Time (s)')
    plt.title('Rewarod Post Process')

