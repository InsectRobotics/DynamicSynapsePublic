# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 20:28:36 2017

@author: chitianqilin
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:18:49 2017

@author: chitianqilin
"""
#import DynamicSynapseArray2DLimitedDiffuse as DSA
#import DynamicSynapseArray2D as DSA
import DynamicSynapseArray2D20180103 as DSA
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import platform
import shutil


class StepAngleSequenceAnalyzer:
    def __init__(self,condition):
        self.PostiveExploring = 0
        self.NegativeExploring = 0
        self.PostiveTurnNumber = 0
        self.NegativeTurnNumber = 0
        self.ThroughUpThreshold = 0
        self.ThroughDownThreshold = 0
        self.PostiveLargeTurnNumber = 0
        self.NegativeLargeTurnNumber = 0
        self.MaxPolarAngle = 0
        self.MinPolarAngle = 0
        self.ZeroCrossFind = 0
        self.timeConditionFit = 0
        self.timeConditionNotFit = 0
        self.timeLast = 0
        self.justStart = 1
        self.condition=condition
        self.MaxPolarAngleArray =[]
        self.MinPolarAngleArray = []
        self.ZeroCrossTimeArray = []
        self.timeConditionFitArray = []
        self.timeConditionNotFitArray = []
    def StepAnalyze(self, CurrentTime, CurrentAngle):
        self.ZeroCrossFind = 0
        if self.justStart == 1:
            if CurrentAngle <0:
                self.NegativeExploring = 1
                self.justStart = 0
            elif CurrentAngle > 0:
                self.PostiveExploring = 1  
                self.justStart = 0
        else:
            if self.NegativeExploring == 1:
                if CurrentAngle < -self.condition:
                    self.ThroughDownThreshold = 1
                if CurrentAngle < self.MinPolarAngle:
                    self.MinPolarAngle = CurrentAngle
                if CurrentAngle > 0:
                    self.ZeroCrossFind = 1
            elif self.PostiveExploring == 1:
                if CurrentAngle > self.condition:
                    self.ThroughUpThreshold = 1
                if CurrentAngle > self.MaxPolarAngle:
                    self.MaxPolarAngle = CurrentAngle
                if CurrentAngle < 0:
                    self.ZeroCrossFind = 1
        
        if self.ZeroCrossFind == 1:
            self.ZeroCrossTimeArray.append(CurrentTime)
            if self.NegativeExploring == 1:
                self.NegativeTurnNumber += 1
                if self.ThroughDownThreshold == 1:
                    self.NegativeLargeTurnNumber += 1
                    self.timeConditionFitArray.append( CurrentTime-self.timeLast)
                    self.timeConditionFit += self.timeConditionFitArray[-1]
                    self.ThroughDownThreshold = 0
                else:
                    self.timeConditionNotFitArray.append( CurrentTime-self.timeLast)
                    self.timeConditionNotFit += self.timeConditionNotFitArray[-1]
                self.NegativeExploring = 0
                self.PostiveExploring = 1
                self.MinPolarAngleArray.append(self.MinPolarAngle)
                self.MinPolarAngle = 0
            elif self.PostiveExploring == 1: 
                self.PostiveTurnNumber += 1
                if self.ThroughUpThreshold == 1:
                    self.PostiveLargeTurnNumber += 1
                    self.timeConditionFitArray.append( CurrentTime-self.timeLast)
                    self.timeConditionFit += self.timeConditionFitArray[-1]
                    self.ThroughUpThreshold = 0                      
                else:
                    self.timeConditionNotFitArray.append( CurrentTime-self.timeLast)
                    self.timeConditionNotFit += self.timeConditionNotFitArray[-1]
                self.NegativeExploring = 1
                self.PostiveExploring = 0 
                self.MaxPolarAngleArray.append(self.MaxPolarAngle)
                self.MaxPolarAngle = 0
            self.timeLast = CurrentTime
            
            
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

class CPG:
    
    def __init__(self, DS=False, Recording=False, RecordLenth=1000,NumberOfInputs=0,dt=30):
        self.NumberOfNeurons=2
        self.FirigRate = np.array([-1, 1]) #np.ones(2) + 0.1 * (np.random.rand(2)-0.5)
        self.FirigRateLast = np.ones(2)
        self.FirigRateVar = np.zeros(2)
        self.FirigRateVarLast = np.zeros(2)
        self.SynapticWeightsBase = np.array([[0.0001,-0.0001], [-0.0001, 0.0001]])
        self.SynapticWeights = self.SynapticWeightsBase
        self.E = 1
        self.DS=DS
        self.RecordingFlag=Recording
        self.RecordLenth=RecordLenth
        self.NumberOfInputs=NumberOfInputs
        self.dt = dt
        self.t = 0
        if self.DS:
            self.InitDS(NumberOfInputs,dt)
        if Recording:
            self.InitRecording(RecordLenth)
        
    def Derivitive(self):
        DFirigRateVar = 0.4*((-0.001* FirigRateVarLast*(FirigRateVarLast**2 + FirigRateLast**2 -E)/E-np.dot(SynapticWeights *(10 ** (2*(ADSA.Weighters[0, 0:2]-0.5))), FirigRateLast))*dt)  
        DFirigRate = 0.4*((FirigRateVar)*dt)  #

    def Step(self, dt, inputs, ModulatorAmount=0, Weighters=None):
        self.t += self.dt
        if Weighters is None and self.DS:
            Weighters=self.ADSA.Weighters
        if Weighters is not None or self.DS:
            self.SynapticWeights = np.hstack((self.SynapticWeightsBase *(10 ** (2*(Weighters[:,:self.NumberOfNeurons]-0.5))),Weighters[:,self.NumberOfNeurons:]))
        self.FirigRateVar = 0.4*((-0.001* self.FirigRateVarLast*(self.FirigRateVarLast**2 + self.FirigRateLast**2 -self.E)/self.E-np.dot(self.SynapticWeights , np.hstack((self.FirigRateLast, inputs)) ))*dt) + self.FirigRateVarLast 
        self.FirigRate = 0.4*((self.FirigRateVar)*dt) + self.FirigRateLast   #
        if self.DS:
            self.ADSA.StepSynapseDynamics(dt, ModulatorAmount)
            
    def StateUpdate(self):
        self.FirigRateLast = self.FirigRate
        self.FirigRateVarLast = self.FirigRateVar
        if self.DS:
            self.ADSA.StateUpdate()
            
    def InitRecording(self, lenth):
        self.RecordingState = True
        self.RecordingLenth = lenth
        self.RecordingInPointer = 0
        self.Trace = {'FirigRate':np.zeros(np.append(lenth,self.NumberOfNeurons).ravel()), \
                        'FirigRateVar' : np.zeros(np.append(lenth,self.NumberOfNeurons).ravel()),\
                        't':np.zeros(np.append(lenth,self.NumberOfNeurons).ravel())
                        }
        if self.DS:
            self.ADSA.InitRecording(lenth)
    def Recording(self, N_Append=100):
        Temp = None
        for key in self.Trace:
            exec("Temp = self.%s" % (key))
#            print ("Temp = self.%s" % (key))
#            print(Temp)
            self.Trace[key][self.RecordingInPointer, :] = Temp
        self.RecordingInPointer += 1
        if self.RecordingInPointer>= self.RecordingLenth:
            if N_Append == 0:
                self.RecordingInPointer = 0
            else:
                for key in self.Trace:
                    self.Trace[key]=np.append(self.Trace[key],np.zeros(np.append(N_Append,self.NumberOfNeurons).ravel()), axis=0)
        if self.DS:
            self.ADSA.Recording()
    def RecordingFinish(self):
        for key in self.Trace:
            self.Trace[key]=np.delete(self.Trace[key], np.s_[self.RecordingInPointer::], axis=0)
        if self.DS:
            self.ADSA.RecordingFinish()
            
    def InitDS(self, NumberOfInputs,dt):
        NumberOfSynapses = np.array([self.NumberOfNeurons, 2+NumberOfInputs])
        Weighters = 0.1*(np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])-0.5) + 0.2 * np.ones([NumberOfSynapses[0], NumberOfSynapses[1]]) # 0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)
        WeighteAmountPerSynapse = 1
        WeighterVarRates = np.zeros(NumberOfSynapses)
        tauWV =0.5#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
        aWV = 100#170#130   100
        rWV = 7000*500#7000
        scale=1
        damping =2*7000
        CW = 100
        WeightersCentre = np.ones(NumberOfSynapses) * 0.3 + 0.1 * np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])   # np.ones(NumberOfSynapses)/2+[0.2, 0.1,0.0, 0.2, 0]  #[0.2, 0.1, 0]
        WeighterVarDamping = np.ones(NumberOfSynapses) * damping  # [10,2,2,2,2]       #[2,2,2]
        WeightersCentreUpdateRate = 0.000012*10
        DampingUpdateRate = 0.0000002/100/350
        
        self.ADSA = DSA.DynamicSynapseArray(NumberOfSynapses=NumberOfSynapses, CW=CW, tauWV=tauWV, aWV=aWV, rWV=rWV, scale=scale,
                                       WeightersCentre=WeightersCentre, WeighterVarDamping=WeighterVarDamping, WeighteAmountPerSynapse=WeighteAmountPerSynapse,
                                       Weighters=Weighters, WeighterVarRates=WeighterVarRates, WeightersCentreUpdateRate=WeightersCentreUpdateRate, DampingUpdateRate = DampingUpdateRate)
        
    def Plot(self, PlotDownSampleRate=100):
        labels = [str(i) for i in range(self.NumberOfNeurons)]
        figure9 = plt.figure()
        plt.plot(self.Trace['t'][::PlotDownSampleRate], self.Trace['FirigRate'][::PlotDownSampleRate])
        plt.title('FirigRate')
        figure10 = plt.figure()
        figure10lines = plt.plot(self.Trace['t'][::PlotDownSampleRate], self.Trace['FirigRateVar'][::PlotDownSampleRate], label=labels)
        plt.legend(figure10lines, labels)
        plt.title('FirigRateVar')    
        figure12 = plt.figure()
        plt.plot(self.Trace['FirigRate'][:,0], self.Trace['FirigRateVar'][:,0])
        plt.title('FirigRate x to FirigRateVar y') 
        figure13 = plt.figure()
        plt.plot(self.Trace['FirigRate'][:,0], self.Trace['FirigRate'][:,1])
        plt.title('FirigRate x to FirigRate y') 
        figure14, ax14 = plt.subplots(2 )
        ax14[0].plot(self.Trace['t'][1000:2000], self.Trace['FirigRate'][1000:2000])
        plt.title('FirigRate begining')
        ax14[1].plot(self.Trace['t'][-1000:], self.Trace['FirigRate'][-1000:])
        plt.title('FirigRate ending')        
        return [figure9,figure10,figure12,figure13,figure14]
if __name__ == "__main__":

    # %%
    path=ChoosePath()
    TimeOfRecording = time.time()
    TimeOfRecordingStr=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    print('Time of the start of Simulation'+TimeOfRecordingStr)
        #%%
    SimulationResultPath = path+'CPGTest'+TimeOfRecordingStr+'/'
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
    Weighters = 0.1*(np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])-0.5) + 0.2 * np.ones([NumberOfSynapses[0], NumberOfSynapses[1]]) # 0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)
    WeighteAmountPerSynapse = 1
#    WeighterInAxon = WeighteAmountPerSynapse* NumberOfSynapses - Weighters.sum()
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
#    WeightersCentre = np.ones(NumberOfSynapses) * 0.2 + 0.1 * np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1])   # np.ones(NumberOfSynapses)/2+[0.2, 0.1,0.0, 0.2, 0]  #[0.2, 0.1, 0]
    WeightersCentre = np.ones(NumberOfSynapses) * 0.3 + 0.2 * np.random.rand(NumberOfSynapses[0], NumberOfSynapses[1]) 
    WeighterVarDamping = np.ones(NumberOfSynapses) * damping  # [10,2,2,2,2]       #[2,2,2]
#    Parameters = [CW, tauWV, aWV , rWV, scale, WeightersCentre,WeighterVarDamping]
    WeightersCentreUpdateRate = 0.0002 #0.00012
    DampingUpdateRate = 0.0000000002
    
    ADSA = DSA.DynamicSynapseArray(NumberOfSynapses=NumberOfSynapses, CW=CW, tauWV=tauWV, aWV=aWV, rWV=rWV, scale=scale,
                                   WeightersCentre=WeightersCentre, WeighterVarDamping=WeighterVarDamping, WeighteAmountPerSynapse=WeighteAmountPerSynapse,
                                   Weighters=Weighters, WeighterVarRates=WeighterVarRates, WeightersCentreUpdateRate=WeightersCentreUpdateRate, DampingUpdateRate = DampingUpdateRate)
    
    SASA = StepAngleSequenceAnalyzer(0.5)
        # %%
    FirigRate = np.array([-1, 1]) #np.ones(2) + 0.1 * (np.random.rand(2)-0.5)
    FirigRateLast = np.ones(2)
    FirigRateVar = np.zeros(2)
    FirigRateVarLast = np.zeros(2)
    SynapticWeights = np.array([[0.0001,-0.0001], [-0.0001, 0.0001]])

    TimeAccu = 0
    T = 1000000
    dt = 10
    steps = np.int64(np.floor(T/dt))
    t = np.linspace(0., T, steps)
    ADSA.InitRecording(lenth=steps)
    E=1
    TargetPeriod = 1000
    ModulatorAmount = 0
    Error = 0
    ErrorLast = 0
    
    Tracet = t
    TraceModulatorAmount = np.zeros(steps)
    TraceFirigRate = np.zeros([steps, 2])
    TraceFirigRateVar = np.zeros([steps, 2])
    TraceError = np.zeros(steps)
    PlotDownSampleRate = 100
    for step in range(steps):  
          
        FirigRateLast = FirigRate
        FirigRateVarLast = FirigRateVar
        ErrorLast = Error 
        SASA.StepAnalyze(t[step], FirigRateLast[0])
        ADSA.StateUpdate()
        
        FirigRateVar = 0.4*((-0.001* FirigRateVarLast*(FirigRateVarLast**2 + FirigRateLast**2 -E)/E-np.dot(SynapticWeights *(10 ** (2*(ADSA.Weighters[0, 0:2]-0.5))), FirigRateLast))*dt) + FirigRateVarLast 
        FirigRate = 0.4*((FirigRateVar)*dt) + FirigRateLast   #
        
        if SASA.ZeroCrossFind and len (SASA.ZeroCrossTimeArray) >=3:
            Error = (SASA.ZeroCrossTimeArray[-1] - SASA.ZeroCrossTimeArray[-2]) - TargetPeriod
            ModulatorAmount =  (abs(ErrorLast)-abs(Error))/10#10
            if ModulatorAmount < 0:
                  ModulatorAmount = 0
#        else:
#            ModulatorAmount = 0
       # ModulatorAmount = 0
        ADSA.StepSynapseDynamics(dt, ModulatorAmount)
        ADSA.Recording()    
            
        TraceFirigRate[step] = FirigRate 
        TraceFirigRateVar[step] = FirigRateVar
        TraceModulatorAmount[step] = ModulatorAmount
        TraceError[step] = Error
        
        if step != 0 and step%(steps/100) <1:
              PresentTime = time.time()
              PresentTimeStr = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
              TimeEplapsed = PresentTime - TimeOfRecording
              percent = step*100/steps
              TimeRemain = TimeEplapsed/percent*(100-percent)
              print ('step %d, %d  completed at '%(step, percent) + PresentTimeStr, ' Time left: ' + str(TimeRemain))
# %%
    
    Trace = [Tracet, ADSA.Trace['Weighters'][:, 0, :], ADSA.Trace['WeighterVarRates'][:, 0, :], ADSA.Trace['WeighterInDendriteConcentration'][:, 0], ADSA.Trace['WeightersCentre'][:, 0, :], ADSA.Trace['WeighterVarDamping'][:, 0, :], ADSA.Trace['EquivalentVolume'][:, 0, :]]
    #figure1,figure2,figure3,figure4,figure5, figure6, figure7, ax 
    FigureDict, ax= DSA.plot(TimeOfRecordingStr, Trace, StartTimeRate=0, DownSampleRate=PlotDownSampleRate)
#%% 
    labels = [str(i) for i in range(NumberOfSynapses[1])]
    figure8 =  plt.figure()
    plt.plot(Tracet[::PlotDownSampleRate], TraceModulatorAmount[::PlotDownSampleRate])
    plt.title('ModulatorAmount')
    figure9 = plt.figure()
    plt.plot(Tracet[::PlotDownSampleRate], TraceFirigRate[::PlotDownSampleRate])
    plt.title('FirigRate')
    figure10 = plt.figure()
    figure10lines = plt.plot(Tracet[::PlotDownSampleRate], TraceFirigRateVar[::PlotDownSampleRate], label=labels)
    plt.legend(figure10lines, labels)
    plt.title('FirigRateVar')    
    figure11 = plt.figure()
    figure11lines = plt.plot(Tracet[::PlotDownSampleRate], TraceError[::PlotDownSampleRate])
    plt.title('Error')       
    figure12 = plt.figure()
    plt.plot(TraceFirigRate[:,0], TraceFirigRateVar[:,0])
    plt.title('FirigRate x to FirigRateVar y') 
    figure13 = plt.figure()
    plt.plot(TraceFirigRate[:,0], TraceFirigRate[:,1])
    plt.title('FirigRate x to FirigRate y') 
    figure14, ax14 = plt.subplots(2 )
    ax14[0].plot(Tracet[1000:2000], TraceFirigRate[1000:2000])
    plt.title('FirigRate begining')
    ax14[1].plot(Tracet[-1000:], TraceFirigRate[-1000:])
    plt.title('FirigRate ending')
    
    figure8.tight_layout()
    figure9.tight_layout()
    figure10.tight_layout()
    figure11.tight_layout()
    figure12.tight_layout()
    figure13.tight_layout()
    figure14.tight_layout()
 
# %%
    ppName = "CPGTestFigures"+TimeOfRecordingStr
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
    figure11.savefig(pp, format='pdf')
    figure12.savefig(pp, format='pdf')
    figure13.savefig(pp, format='pdf')
    figure14.savefig(pp, format='pdf')
    figure15.savefig(pp, format='pdf')
    figure16.savefig(pp, format='pdf')
    figure17.savefig(pp, format='pdf')

    pp.close()
    TimeOfEndOfSimulation=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('Time of the End of Simulation'+TimeOfEndOfSimulation)