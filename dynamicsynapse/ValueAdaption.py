# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 12:31:37 2018

@author: chitianqilin
"""

        NeuronSensitivity = np.ones(num_actions*2)*0.5
        NeuronSensitivityUpdateRate = 0.000001
        NeuronSensitivity2 = np.ones(num_actions)*0.005
        NeuronSensitivityUpdateRate2 = 0.000001
        
        NegativeReward=-1* np.ones(NumberOfSubSys)
        PostiveReward=1* np.ones(NumberOfSubSys)
        rewardSensitivity=1* np.ones(NumberOfSubSys)
        RewardBias=np.zeros(NumberOfSubSys)
        ModulatorSensitivityUpdateRate= 0.000001
        PRwdDecayRate = 0.00001
        NRwdDecayRate  = 0.00001
