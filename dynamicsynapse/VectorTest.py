#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:18:49 2017

@author: chitianqilin
"""

import DynamicSynapseArray2D20180103 as DSA
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import platform
import shutil

def ChoosePath():
    if platform.node()=='balder-HP-Z640-Workstation':
        path="/home/chitianqilin/Documents/SimulationResult/DynamicSynapse/"
    elif platform.node()=='valhalla-ws':
        path="/home/chitianqilin/SimulationResult/DynamicSynapse/"
    elif platform.node()=='chitianqilin-hp':
        path="I:/OneDrive - University of Edinburgh/sharedwithUbuntu/python/SimulationResult/DynamicSynapse/"
    elif platform.node()=='chitianqilin-LT' or platform.node()=='chitianqilin-HM' :
        path="/home/chitianqilin/SimulationResult/DynamicSynapse/"
    else:
        path=''
    if not os.path.exists(path):
        os.makedirs(path)        
    return path

if __name__ == "__main__":

    # %%
    path=ChoosePath()
    TimeOfRecording = time.time()
    TimeOfRecordingStr=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    print('Time of the start of Simulation'+TimeOfRecordingStr)
        #%%
    SimulationResultPath = path+'VectorTest'+TimeOfRecordingStr+'/'
    codepath = SimulationResultPath+'src/'
    plotpath = SimulationResultPath+'plot/'
    #%%
    if not os.path.exists(codepath):
        os.makedirs(codepath)   
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".py"): 
            shutil.copy2(filename, codepath)
            
    if not os.path.exists(plotpath):
        os.makedirs(plotpath) 
            # %%
    NumberOfSynapses = np.array([2, 6])
    Weighters = 0.2*(np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])-0.5) + 0.3 * np.ones([NumberOfSynapses[0], NumberOfSynapses[1]]) # 0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)
    WeighteAmountPerSynapse = 1
#    WeighterInDendrite = WeighteAmountPerSynapse* NumberOfSynapses - Weighters.sum()
    WeighterVarRates = np.zeros(NumberOfSynapses)

#    CW = 100
#    tauWV = 40
#    aWV = 100
#    rWV = 5000# 5000
#    scale = 1
    
    tauWV =0.5#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
    aWV = 100#170#130   100
    rWV = 7000*500#7000
    scale=1
    damping =2*7000
    CW = 100
   
    WeightersCentre = np.ones(NumberOfSynapses) * 0.3+ 0.001 * (np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])-0.5)   # np.ones(NumberOfSynapses)/2+[0.2, 0.1,0.0, 0.2, 0]  #[0.2, 0.1, 0]
    WeighterVarDamping = np.ones(NumberOfSynapses) * damping#2  # [10,2,2,2,2]       #[2,2,2]
#    Parameters = [CW, tauWV, aWV , rWV, scale, WeightersCentre,WeighterVarDamping]
    WeightersCentreUpdateRate = 0.0001
    DampingUpdateRate = 0.0000000001
    
    ADSA = DSA.DynamicSynapseArray(NumberOfSynapses=NumberOfSynapses, CW=CW, tauWV=tauWV, aWV=aWV, rWV=rWV, scale=scale,
                                   WeightersCentre=WeightersCentre, WeighterVarDamping=WeighterVarDamping, WeighteAmountPerSynapse=WeighteAmountPerSynapse,
                                   Weighters=Weighters, WeighterVarRates=WeighterVarRates, WeightersCentreUpdateRate=WeightersCentreUpdateRate, DampingUpdateRate = DampingUpdateRate)
    
    SensoryInputOriginal = np.arange(6)
    OutputValue = 0
    OutputValueLast = 0
    # T = 100000000
    TimeAccu = 0
    T = 5*60*60*1000#20000000
    dt = 20
    steps = np.int64(np.floor(T/dt))
    t = np.linspace(0., T, steps)
    ADSA.InitRecording(lenth=steps)
    Tracet = t
    TraceModulatorAmount = np.zeros(steps)
    TraceOutputValue = np.zeros(steps)
    TraceSensoryInput = np.zeros([steps, NumberOfSynapses[1]])
    OutputValueLPF = 0
    OutputValueLPFUpdateRate = 0.00001
    for step in range(steps):  
          #TimeAccu += dt
          OutputValueLast = OutputValue
          ADSA.StateUpdate()
          SensoryInputNoisy = SensoryInputOriginal + ADSA.Weighters[1,:]/5
          SensoryInput = SensoryInputNoisy 
          OutputValue = np.sum(SensoryInput * ADSA.Weighters[0,:])
#          OutputValueLPF=(OutputValueLPF*(1-OutputValueLPFUpdateRate)+OutputValue*OutputValueLPFUpdateRate)*dt
          ModulatorAmount = 10* ((OutputValue - OutputValueLast) * (OutputValue-4.5) if OutputValue - OutputValueLast>0 and OutputValue > 5 else 0) #
          if step <2000:
              ModulatorAmount = 0
          ModulatorAmountArray=(np.ones(NumberOfSynapses).T*np.array([ModulatorAmount,0])).T
          ADSA.StepSynapseDynamics(dt, ModulatorAmountArray, Compensate="Dynamic1")
          ADSA.Recording()
          TraceModulatorAmount[step] = ModulatorAmount
          TraceOutputValue[step] = OutputValue
          TraceSensoryInput[step] = SensoryInput
          if step != 0 and step%(steps/100) <1:
                PresentTime = time.time()
                PresentTimeStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                TimeEplapsed = PresentTime - TimeOfRecording
                percent = step*100/steps
                TimeRemain = TimeEplapsed/percent*(100-percent)
                print ('step %d, %d  completed at '%(step, percent) + PresentTimeStr, ' Time left: ' + str(TimeRemain))
# %%
    
    Trace = [Tracet, ADSA.Trace['Weighters'][:, 0, :], ADSA.Trace['WeighterVarRates'][:, 0, :], ADSA.Trace['WeighterInDendriteConcentration'][:, 0], ADSA.Trace['WeightersCentre'][:, 0, :], ADSA.Trace['WeighterVarDamping'][:, 0, :], ADSA.Trace['EquivalentVolume'][:, 0, :]] 
    FigureDict, ax = DSA.plot(TimeOfRecordingStr, Trace, StartTimeRate=0, DownSampleRate=100)
    labels = [str(i) for i in range(NumberOfSynapses[1])]
    figure8 =  plt.figure()
    plt.plot(Tracet[::100], TraceModulatorAmount[::100])
    plt.title('ModulatorAmount')
    figure9 = plt.figure()
    plt.plot(Tracet[::100], TraceOutputValue[::100])
    plt.title('OutputValue')
    figure10 = plt.figure()
    figure10lines = plt.plot(Tracet[::100], TraceSensoryInput[::100], label=labels)
    plt.legend(figure10lines, labels)
    plt.title('SensoryInput')    
      

    
    figure8.tight_layout()
    figure9.tight_layout()
    figure10.tight_layout()
# %%
    ppName = "VectorTestFigures"+TimeOfRecordingStr
    pp = PdfPages(plotpath+ppName + '.pdf')
    figure1.savefig(pp, format='pdf') 
    figure2.savefig(pp, format='pdf') 
    figure3.savefig(pp, format='pdf') 
    figure4.savefig(pp, format='pdf') 
    figure5.savefig(pp, format='pdf') 
    figure6.savefig(pp, format='pdf') 
    figure7.savefig(pp, format='pdf')
    figure8.savefig(pp, format='pdf') 
    figure9.savefig(pp, format='pdf')
    figure10.savefig(pp, format='pdf')
    pp.close()
    TimeOfEndOfSimulation=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('Time of the End of Simulation'+TimeOfEndOfSimulation)