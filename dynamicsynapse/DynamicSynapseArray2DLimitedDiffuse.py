#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:45:33 2017

@author: chitianqilin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:53:42 2017

@author: chitianqilin
"""

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import dill
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages 
from multiprocessing import Pool, TimeoutError
import logging, sys,  traceback
from cycler import cycler
def rk4(h, y, inputs, Parameters, f):
        k1 = f(y, inputs, Parameters)
#        print(y)
#        print(h)
#        print(k1)
        k2 = f(y + 0.5*h*k1, inputs, Parameters)
        k3 = f(y + 0.5*h*k2, inputs, Parameters)
        k4 = f(y + k3*h, inputs, Parameters)
        return y + (k1 + 2*(k2 + k3) + k4)*h/6.0
    

        
class DynamicSynapseArray:
    def __init__(self, NumberOfSynapses = 5, CW = 100, tauWV = 40, aWV = 100, rWV = 5000, scale=10,  \
                WeightersCentre = None , WeighterVarDamping = None, WeighteAmountPerSynapse = 1, \
                Weighters = None, WeighterVarRates = None, WeightersCentreUpdateRate = 0.000012,\
                DampingUpdateRate = 0.0000003/100/350 , WeightersCentreUpdateCompensate =0,\
                MaxDamping = 10*7000, DiffuseRateInDendrite=0.0001):
        #self.NumberOfNeuron=NumberOfSynapses[0]
        self.NumberOfSynapses = NumberOfSynapses#[1]
        self.CW = CW
        self.tauWV = tauWV
        self.aWV = aWV 
        self.rWV = rWV
        self.scale =scale
        
        self.WeightersCentre = WeightersCentre if WeightersCentre is not None else np.ones(NumberOfSynapses)/2+[[0.1, 0.1,0.1, 0.1, 0.1]]*NumberOfSynapses[0] #[0.2, 0.1, 0]
        self.WeighterVarDamping = WeighterVarDamping  if WeighterVarDamping is not None else  np.ones(NumberOfSynapses) * [[2,2,2,2,2]]*NumberOfSynapses[0] #[2,2,2]
        self.DampingUpdateRate = DampingUpdateRate
        self.GetParameters = lambda: [self.CW, self.tauWV, self.aWV , self.rWV, self.scale, self.WeightersCentre,self.WeighterVarDamping]
        self.Parameters = [self.CW, self.tauWV, self.aWV , self.rWV, self.scale, self.WeightersCentre,self.WeighterVarDamping]
        self.WeighteAmountPerSynapse = WeighteAmountPerSynapse
        
        
        self.WeightersLast = Weighters if Weighters is not None else 0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)
        self.Weighters = self.WeightersLast
        self.WeighterInDendrite =  self.WeighteAmountPerSynapse*np.ones(NumberOfSynapses) - self.WeightersLast
        self.WeighterInDendriteConcentration = self.WeighterInDendrite
        self.WeighterInDendriteLast =  self.WeighteAmountPerSynapse*np.ones(NumberOfSynapses) - self.WeightersLast
        self.WeighterInDendriteConcentrationLast = self.WeighterInDendrite
        self.DiffuseRateInDendrite = DiffuseRateInDendrite

        self.WeighterVarRatesLast = WeighterVarRates if WeighterVarRates is not None else np.zeros(NumberOfSynapses)
        self.WeighterVarRates = self.WeighterVarRatesLast
        
        self.EquivalentVolume = (1+(2*self.WeightersCentre-(self.WeighterInDendriteConcentrationLast+self.WeightersLast))/((self.WeighterInDendriteConcentrationLast+self.WeightersLast)-self.WeightersCentre))

        self.WeightersCentreUpdateRate = WeightersCentreUpdateRate
        self.WeightersCentreUpdateCompensate = WeightersCentreUpdateCompensate
        
        self.MaxDamping = MaxDamping
        
        self.OsciBias = np.greater(self.Weighters, 0)
        self.LastOsciBias = np.greater(self.Weighters, 0)
        self.DownWInterg = np.ones_like(self.Weighters)*4000
        self.DownWIntergLast = np.ones_like(self.Weighters)*4000
        self.UpWInterg = np.ones_like(self.Weighters)*4000
        self.UpWIntergLast = np.ones_like(self.Weighters)*4000 
        self.t = 0
        self.dt = 10
        
        self.ModulatorAmount = 0
    def Derivative (self, state=None , inputs=None, Parameters=None):
        if state is not None:
            WeightersLast, WeighterVarRatesLast, WeighterInDendriteLast, WeighterInDendriteConcentrationLast = state
        else:
            WeightersLast, WeighterVarRatesLast, WeighterInDendriteLast, WeighterInDendriteConcentrationLast = self.WeightersLast, self.WeighterVarRatesLast, self.WeighterInDendriteLast, self.WeighterInDendriteConcentrationLast
#        if inputs is not None:
#            WeighterInDendriteConcentrationLast=inputs
#        else:
#            WeighterInDendriteConcentrationLast=self.WeighterInDendriteConcentrationLast
        if Parameters is not None:
            CW, tauWV, aWV, rWV, scale, WeightersCentre,WeighterVarDamping = Parameters
        else:
            CW, tauWV, aWV, rWV, scale,  WeightersCentre,WeighterVarDamping = self.Parameters()
        DiffuseRateInDendrite = self.DiffuseRateInDendrite
        #print(WeighterVarRatesLast , WeighterInDendriteConcentration , WeightersLast , self.scale)
        EquivalentVolume = (1+(2*WeightersCentre-(WeighterInDendriteConcentrationLast+WeightersLast))/((WeighterInDendriteConcentrationLast+WeightersLast)-WeightersCentre))
        self.EquivalentVolume = EquivalentVolume
#        DWeighters =  WeighterVarRatesLast * WeighterInDendriteConcentration * WeightersLast /self.scale*2
#        DWeighters =  WeighterVarRatesLast /self.scale/2
#        CW = 100
#        tauWV = 17.8#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100 
#        rWV = 5000
#        scale=1
#        tauWV =500#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#130   100
#        rWV = 7000 #7000
#        scale=1
#        damping =2
#        CW = 100
#        SimulationTimeInterval = 10
 #       DWeighters =  WeighterVarRatesLast  * ( WeighterInDendriteConcentration + WeightersLast/EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInDendriteConcentration - WeightersLast/self.EquivalentVolume)) /2  /self.scale    
#Original
        DWeighters =  WeighterVarRatesLast  * ( WeighterInDendriteConcentrationLast + WeightersLast/EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInDendriteConcentrationLast - WeightersLast/self.EquivalentVolume)) /2  /self.scale    
#        DWeighters = EquivalentVolume *  WeighterVarRatesLast  /self.scale
#        DWeighters =  (WeighterVarRatesLast  *0.5*(1-np.exp( -10*( WeighterInDendriteConcentration + WeightersLast/EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInDendriteConcentration - WeightersLast/EquivalentVolume)) /2 )))/self.scale   
#        DWeighters =  WeighterVarRatesLast * WeighterInDendriteConcentration * WeightersLast/scale
        DWeighterVarRates = (  (tauWV*(WeighterInDendriteConcentrationLast - WeightersLast/EquivalentVolume ) \
        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5)) - WeighterVarDamping*WeighterVarRatesLast )  /scale / rWV
#        print(DWeighters)
#        print(DWeighterVarRates)                        

##        DWeighters =  WeighterVarRatesLast * WeighterInDendriteConcentration * WeightersLast /self.scale*2
##        DWeighters =  WeighterVarRatesLast /self.scale/2
##        CW = 100
##        tauWV = 17.8#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
##        aWV = 100 
##        rWV = 5000
##        scale=1
#        DWeighters =  WeighterVarRatesLast  * ( WeighterInDendriteConcentration + WeightersLast/self.EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInDendriteConcentration - WeightersLast/self.EquivalentVolume)) /2  /self.scale    
#        DWeighterVarRates = (  (WeighterInDendriteConcentration - WeightersLast/self.EquivalentVolume  \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /self.scale 

#very chaos, no distinguish between pump rate and transport speed
#        CW = 100
#        tauWV = 17.8--50#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100 
#        rWV = 5000
#        scale=1
#        DWeighters =  WeighterVarRatesLast /self.scale
#        DWeighterVarRates = (  (WeighterInDendriteConcentration - WeightersLast/self.EquivalentVolume  \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV * ( WeighterInDendriteConcentration + WeightersLast +np.sign(WeighterVarRatesLast)*(WeighterInDendriteConcentration - WeightersLast)) /2  /self.scale
#instantious catch-up lateral mobility resistance, no pump resistance
##        CW = 100
##        tauWV = 40#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
##        aWV = 0.0005#0.02 #100 
##        rWV = 200000
##        scale=1
#        DWeighters =  WeighterVarRatesLast +  (WeighterInDendriteConcentration - WeightersLast/self.EquivalentVolume)/ rWV /self.scale
#        DWeighterVarRates = (  \
#        (aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5) - WeighterVarDamping*WeighterVarRatesLast )  * ( WeighterInDendriteConcentration + WeightersLast +np.sign(WeighterVarRatesLast)*(WeighterInDendriteConcentration - WeightersLast)) /2 )/self.scale

#original
#        DWeighters =  WeighterVarRatesLast * WeighterInDendriteConcentration * WeightersLast /self.scale
#        DWeighterVarRates = (  (WeighterInDendriteConcentration - WeightersLast/(1+(2*WeightersCentre-(WeighterInDendriteConcentration+WeightersLast))/((WeighterInDendriteConcentration+WeightersLast)-WeightersCentre) ) \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV  - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /self.scale
        DiffuseLR=(WeighterInDendriteConcentrationLast[:,:-1]-WeighterInDendriteConcentrationLast[:,1:])*DiffuseRateInDendrite
        DiffuseFromR=np.hstack((-DiffuseLR,np.zeros((WeighterInDendriteConcentrationLast.shape[0],1))))
        DiffuseFromL=np.hstack((np.zeros((WeighterInDendriteConcentrationLast.shape[0],1)),DiffuseLR))
        DWeighterInDendrite = DiffuseFromR+DiffuseFromL-DWeighters
        WeighterInDendriteConcentrationLast = DWeighterInDendrite
        return np.array([DWeighters, DWeighterVarRates, DWeighterInDendrite, WeighterInDendriteConcentrationLast ])
    

    def StepSynapseDynamics(self, dt, ModulatorAmount, Compensate='Constant'):
        if isinstance(ModulatorAmount,(list, tuple,np.ndarray)):
            self.ModulatorAmount=ModulatorAmount
            ModulatorAmount=np.array(ModulatorAmount)
            ModulatorAmount=ModulatorAmount.reshape(np.append(ModulatorAmount.shape,1))
        else:
            self.ModulatorAmount=ModulatorAmount

        if dt is None:
            self.t += self.dt
        else:
            self.t += dt
        self.Parameters = [self.CW, self.tauWV, self.aWV , self.rWV, self.scale, self.WeightersCentre,self.WeighterVarDamping]
        self.Weighters, self.WeighterVarRates, self.WeighterInDendrite, self.WeighterInDendriteConcentration = rk4(dt, np.array([self.WeightersLast, self.WeighterVarRatesLast, self.WeighterInDendriteLast, self.WeighterInDendriteConcentrationLast]), None , self.Parameters, self.Derivative)
        #print(self.Weighters)
        self.WeighterInDendriteConcentration=self.WeighterInDendrite
        if np.isnan(self.Weighters[0,0]):
            pass
#        self.WeighterInDendrite = self.WeighterInDendriteLast- self.WeighteAmountPerSynapse* self.NumberOfSynapses[1] - self.WeightersLast.sum(axis=1,keepdims=True)
#        #print(self.WeighterInDendrite, self.WeighteAmountPerSynapse, self.NumberOfSynapses[1] , self.WeightersLast.sum(axis=1,keepdims=True))
#        self.WeighterInDendriteConcentration = self.WeighterInDendrite/self.NumberOfSynapses[1]
#        if SASA is None:
#            WCUCompen=self.WeightersCentreUpdateCompensate
#        else:
#            for indexI in range(self.Weighters.shape[0]):
#                for indexJ in range(self.Weighters.shape[1]):
#                    SASA[indexI,indexJ].StepAnalyze(step*SimulationTimeInterval, self.Weighters[indexI,indexJ])
#                    if SASA.ZeroCrossFind and len (SASA.ZeroCrossTimeArray) >=3:
#                        Error = (SASA[indexI,indexJ].ZeroCrossTimeArray[-1] - SASA[indexI,indexJ].ZeroCrossTimeArray[-2]
#                    WCUCompen=
        if Compensate == 'Dynamic0':
            self.OsciBias = np.greater(self.Weighters, self.WeightersCentre)
            IndexDownToUp = np.logical_and(np.equal(self.LastOsciBias,False),np.equal(self.OsciBias,True))
#            if np.any(IndexDownToUp):
#                print(IndexDownToUp)
            self.DownWIntergLast[IndexDownToUp]=self.DownWInterg[IndexDownToUp]
            self.DownWInterg[IndexDownToUp]=0
    
            IndexUpToDown = np.logical_and(np.equal(self.LastOsciBias,True),np.equal(self.OsciBias,False))
            self.UpWIntergLast[IndexUpToDown]=self.UpWInterg[IndexUpToDown]
            self.UpWInterg[IndexUpToDown]=0
            
            self.DownWInterg[np.logical_not(self.OsciBias)] +=(self.WeightersCentre[np.logical_not(self.OsciBias)]-self.Weighters[np.logical_not(self.OsciBias)])*dt
            self.UpWInterg[self.OsciBias] +=(self.Weighters[self.OsciBias]-self.WeightersCentre[self.OsciBias])*dt
            
            self.WeightersCentreUpdateCompensate=(self.DownWIntergLast/self.UpWIntergLast)-1
        
            #print(self.WeightersCentreUpdateCompensate)
            self.LastOsciBias = self.OsciBias
        if Compensate == 'Dynamic1':
            self.OsciBias = np.greater(self.Weighters, self.WeightersCentre)
            
            self.DownWInterg[np.logical_not(self.OsciBias)] +=(self.WeightersCentre[np.logical_not(self.OsciBias)]-self.Weighters[np.logical_not(self.OsciBias)])*dt
            self.UpWInterg[self.OsciBias] +=(self.Weighters[self.OsciBias]-self.WeightersCentre[self.OsciBias])*dt
            
            self.WeightersCentreUpdateCompensate=(self.DownWIntergLast/self.UpWIntergLast)-1
           # print(self.WeightersCentreUpdateCompensate)
            self.LastOsciBias = self.OsciBias
            
        self.WeightersCentre += (self.Weighters-self.WeightersCentre)*ModulatorAmount *self.WeightersCentreUpdateRate*dt*(1+self.WeightersCentreUpdateCompensate*np.greater(self.Weighters, self.WeightersCentre)) #0.000015##0.00002
#        self.WeightersCentre += (self.Weighters-self.WeightersCentre)*ModulatorAmount *self.WeightersCentreUpdateRate*dt #0.000015##0.00002
        #print(self.WeightersCentre)
        
        self.WeighterVarDamping += (self.MaxDamping-self.WeighterVarDamping)*self.WeighterVarDamping*ModulatorAmount*self.DampingUpdateRate *dt #
        #print(self.WeighterVarDamping)
        return self.Weighters, self.WeighterVarRates, self.WeighterInDendrite, self.WeighterInDendriteConcentration
        
    def StateUpdate(self):
        self.WeightersLast, self.WeighterVarRatesLast = self.Weighters, self.WeighterVarRates 
        self.WeighterInDendriteLast, self.WeighterInDendriteConcentrationLast = self.WeighterInDendrite, self.WeighterInDendriteConcentration
    def InitRecording(self, lenth, SampleStep = 1):
        self.RecordingState = True
        self.RecordingLenth = lenth
        self.RecordingInPointer = 0
        self.Trace = {'t':np.zeros(np.append(np.append(lenth,self.NumberOfSynapses[0]),1).ravel()),\
                        'Weighters':np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'WeightersCentre' : np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'WeighterVarRates' : np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'WeighterVarDamping' : np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()),\
                        'WeighterInDendriteConcentration' :  np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'EquivalentVolume':np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'ModulatorAmount':np.zeros(np.append(np.append(lenth,self.NumberOfSynapses[0]),1).ravel())
                        }
    def Recording(self, N_Append=0):
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
                    if key == 'ModulatorAmount' or 't':
                        
                        self.Trace[key]=np.append(self.Trace[key],np.zeros(np.append(N_Append,self.Trace[key].shape).ravel()), axis=0)
                    else:
                        self.Trace[key]=np.append(self.Trace[key],np.zeros(np.append(N_Append,self.NumberOfSynapses).ravel()), axis=0)
    def RecordingFinish(self):
        for key in self.Trace:
            self.Trace[key]=np.delete(self.Trace[key], np.s_[self.RecordingInPointer::], axis=0)
#%%                
    def PlotPhasePortrait(self, xlim, ylim, fig=None, ax=None,  inputs=None, Parameters=None):
#        fig2 = plt.figure(figsize=(10, 20))
#        ax4 = fig2.add_subplot(1,1,1)
        if fig == None or ax == None:
            fig, ax = plt.subplots(1,sharex=False , figsize=(20, 12))#

        if Parameters is not None:
            CW, tauWV, aWV, rWV, scale, WeightersCentre,WeighterVarDamping = Parameters
        else:
            CW, tauWV, aWV, rWV, scale, WeightersCentre,WeighterVarDamping = self.GetParameters()
        if inputs is not None:
            d, h  = WeighterInDendriteConcentration, WeightersCentre=inputs
        else:
            d = WeighterInDendriteConcentration=self.WeighterInDendriteConcentration
        a= aWV / tauWV
        b= WeighterVarDamping / tauWV
        h = WeightersCentre
        s=1
        Wis=np.linspace(xlim[0],xlim[1],num=10000)
        w=Wis
        colors=['r','c']
        ax.plot( Wis,(-np.sqrt(h*s*(4*a*d*h*s-4*a*d*s*w+4*a*h*w-4*a*w*w+b**2*h*s))-b*h*s)/(2*a*h*s) , lw=2, label='v-nullcline 0' ) 
        ax.plot( Wis,(np.sqrt(h*s*(4*a*d*h*s-4*a*d*s*w+4*a*h*w-4*a*w*w+b**2*h*s))-b*h*s)/(2*a*h*s) , lw=2, label='v-nullcline 1' ) 
        ax.plot( Wis,(b*h*s-np.sqrt(b**2 * h**2 * s**2 - 4*a*h*s*(d*h*s-d*s*w+h*w-w**2)))/(2*a*h*s), lw=2, label='v-nullcline 2' ) 
        ax.plot( Wis,(b*h*s+np.sqrt(b**2 * h**2 * s**2 - 4*a*h*s*(d*h*s-d*s*w+h*w-w**2)))/(2*a*h*s) , lw=2, label='v-nullcline 3' ) 
        
        ax.plot( Wis,np.zeros(Wis.shape) , lw=2, label='w-nullcline' ) 
        ax.set_ylim (ylim)
#        ax.axvline( 0 , lw=2, label='w-nullcline' ) #
        Wispace=np.linspace(xlim[0],xlim[1], num=30)
        Vspace=np.linspace(ylim[0],ylim[1], num=20)
        Wistep=Wispace[1]-Wispace[0]
        Vstep=Vspace[1]-Vspace[0]

        W1 , V1  = np.meshgrid(Wispace, Vspace) 
        DW1, DV1=self.Derivative(state=[W1, V1] , inputs=d, Parameters=[CW, tauWV, aWV, rWV, scale, WeightersCentre,WeighterVarDamping]  )
#        VectorZ=DW1+ DV1*1j
#        M = np.log(np.hypot(DW1, DV1))
        M = np.greater(DV1, 0)
        ax.quiver(W1 , V1, DW1, DV1, (M), width=0.002, angles='xy')#pivot='mid')



        #ax.legend(bbox_to_anchor=(0.6, 0.2), loc=2, borderaxespad=0.,prop={'size':12})
        ax.legend(prop={'size':12})
        ax.grid()
        return [fig, ax]
    def Plot(self, TimOfRecording, Traces = None, path='', savePlots = False, StartTimeRate=0.3, DownSampleRate=10,linewidth =1, FullScale=False):
        NeuonNumber=0
        newSlice= [slice(None)]*3
        newSlice[1]=NeuonNumber

        if Traces is None:
            Traces = self.Trace['t'][newSlice], self.Trace['Weighters'][newSlice], self.Trace['WeighterVarRates'][newSlice], \
            self.Trace['WeighterInDendriteConcentration'][newSlice], self.Trace['WeightersCentre'][newSlice], \
            self.Trace['WeighterVarDamping'][newSlice], self.Trace['EquivalentVolume'][newSlice]

        self.FigureDict = plot(TimOfRecording, Traces, path, savePlots, StartTimeRate, DownSampleRate,linewidth, FullScale)
        return self.FigureDict
       #%%     
def plot(TimOfRecording, Traces = None, path='', savePlots = False, StartTimeRate=0.3, DownSampleRate=10,linewidth =1, FullScale=False):
#    plt.rc('axes', prop_cycle=(cycler('color',['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','k'])))
    mpl.rcParams['axes.prop_cycle']=cycler('color',['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','b','k'])
#    mpl.rcParams['axes.prop_cycle']=cycler(color='category20')
    if Traces is not None:
        Tracet, TraceWeighters, TraceWeighterVarRates, TraceWeighterInDendriteConcentration, TraceWeightersCentre,TraceWeighterVarDamping,TraceEquivalentVolume = Traces
#        else:
#            for key in self.Trace:
#                exec("Trace%s = self.Trace[%s]" % (key,key))
    TracetInS=Tracet.astype(float)/1000
    NumberOfSteps = len(TracetInS)
    if StartTimeRate == 0:
          StartStep = 0
    else:
          StartStep = NumberOfSteps - int(NumberOfSteps*StartTimeRate)
    
    FigureDict = {}
    figure1 = plt.figure()  
    labels = [str(i) for i in range(TraceWeighters.shape[1])]
    figure1lines = plt.plot(TracetInS, TraceWeighters,  label=labels, linewidth= linewidth)
    plt.legend(figure1lines, labels)
#    figure1lines2 = plt.plot(TracetInS[::DownSampleRate], TraceWeightersCentre[::DownSampleRate], label=labels, linewidth= linewidth)
#    plt.legend(figure1lines2, labels)
    plt.xlabel('Time (s)')
    plt.title('Instantaneous Synaptic Strength')
    FigureDict['Weighters']=figure1
    
    figure2 = plt.figure();
    plt.plot(TracetInS, TraceWeighterVarRates,linewidth= linewidth)
    plt.xlabel('Time (s)')
    plt.title("'Pump' rate")
    FigureDict['PumpRate']=figure2
    
    figure3 = plt.figure()
    ConcentrationLines =plt.plot(TracetInS[::DownSampleRate], TraceWeighters[::DownSampleRate]/TraceEquivalentVolume[::DownSampleRate],linewidth= linewidth)
    plt.legend(ConcentrationLines, labels)
    plt.xlabel('Time (s)')
    plt.title('Receptor Concentration')
    FigureDict['ReceptorConcentration']=figure3
    
    X=TraceWeighters[StartStep:NumberOfSteps,0][::DownSampleRate]
    Y=TraceWeighters[StartStep:NumberOfSteps,1][::DownSampleRate]
    Z=TraceWeighters[StartStep:NumberOfSteps,2][::DownSampleRate]
    
    figure4 = plt.figure()
    plt.plot(X,Y)
    plt.xlabel('Time (s)')
    plt.title('2 Instantaneous Synaptic Strength')
    plt.xlabel('Instantaneous Synaptic Strength 0')
    plt.ylabel('Instantaneous Synaptic Strength 1') 
    FigureDict['2Weighters']=figure4
    
    
    figure5 = plt.figure()
    ax = figure5.add_subplot(111, projection='3d')
    ax.plot(X,Y,Z)
    ax.set_xlabel('Instantaneous Synaptic Strength 0')
    ax.set_ylabel('Instantaneous Synaptic Strength 1')
    ax.set_zlabel('Instantaneous Synaptic Strength 2')
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w',linewidth= linewidth)
    if FullScale:
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)    
    FigureDict['3Weighters']=figure5
    
    figure6 = plt.figure()
    figure6lines = plt.plot(TracetInS[::DownSampleRate], TraceWeightersCentre[::DownSampleRate], label=labels, linewidth= linewidth)
    plt.legend(figure6lines, labels)
    plt.title('Center of Synaptic Strength Oscillation')
    plt.xlabel('Time (s)')
    FigureDict['WeightersCentre']=figure6
    
    figure7 = plt.figure()
    figure7lines = plt.plot(TracetInS[::DownSampleRate], TraceWeighterVarDamping[::DownSampleRate], label=labels, linewidth= linewidth)
    plt.legend(figure7lines, labels)
    plt.title('Damping factor')   
    plt.xlabel('Time (s)')
    FigureDict['DampingFactor']=figure7
    
    figure8 = plt.figure()
    figure8lines = plt.plot(TracetInS[::DownSampleRate], TraceEquivalentVolume[::DownSampleRate], label=labels, linewidth= linewidth)
    plt.legend(figure7lines, labels)
    plt.xlabel('Time (s)')
    plt.title('Receptor Storage Capacity') 
    FigureDict['Capacity']=figure8
    
#%
    figure9 = plt.figure()
    figure9ax1 = figure9 .add_subplot(111)  
    points0,points1 = CrossAnalysis(TraceWeighters[:,0],TraceWeightersCentre[:,0],TraceWeighters,TracetInS)
    if FullScale:
        figure9ax1.set_xlim(0,1)
        figure9ax1.set_ylim(0,1)
#    print('points0')
#    print(points0['points'])
#    print('points1')
#    print(points1['points'])   
    pointsploted0 = figure9ax1.scatter(points0['points'][:,1],points0['points'][:,2],c=points0['t'], cmap=plt.cm.get_cmap('Greens'), marker=".", edgecolor='none') #c=c, ,  cmap=cm
    pointsploted1 = figure9ax1.scatter(points1['points'][:,1],points1['points'][:,2],c=points1['t'], cmap=plt.cm.get_cmap('Blues'), marker=".", edgecolor='none')
    #plt.legend(figure7lines, labels)
    plt.colorbar(pointsploted0)
    plt.colorbar(pointsploted1)
    plt.title('Poincare map') 
    plt.xlabel('Instantaneous Synaptic Strength 1')
    plt.ylabel('Instantaneous Synaptic Strength 2')
    FigureDict['PoincareMap']=figure9
    
    figure10 = plt.figure()
    DendriteConcentrationLines=plt.plot(TracetInS, TraceWeighterInDendriteConcentration,linewidth= linewidth)
    plt.legend(DendriteConcentrationLines, labels,)
    plt.xlabel('Time (s)')
    plt.title('Receptor Concentration in Dendrite')
    FigureDict['ReceptorConcentrationInDendrite']=figure10


 #% 
    for AFig in FigureDict:
        FigureDict[AFig].tight_layout()
    if savePlots == True:
        pp = PdfPages(path+"DynamicSynapse"+TimOfRecording+'.pdf')
        for AFig in FigureDict:
            FigureDict[AFig].savefig(pp, format='pdf')
        pp.close()

            
    return FigureDict, ax
#%%
def CrossAnalysis(Oscillate,Reference,OscillateArray,Tracet):
    points0={'t':[],'points':[]}
    points1={'t':[],'points':[]}
    GreaterThanCentre=(Oscillate[0]>Reference[0])
    print(Oscillate[0])
    print(Reference[0])
    for i1 in range(len(Oscillate)):
        
#        print(Oscillates[i1,0])
#        print(References[i1,0])
        if GreaterThanCentre == True:
            if Oscillate[i1]<Reference[i1]:
                #print(GreaterThanCentre)
                #print(Oscillates[i1,0])
                points0['points'].append(OscillateArray[i1])
                points0['t'].append(Tracet[i1])
                GreaterThanCentre = False
        elif GreaterThanCentre ==  False:
            if Oscillate[i1]>Reference[i1]:
                #print (GreaterThanCentre)
                #print(Oscillates[i1,0])
                points1['points'].append(OscillateArray[i1])
                points1['t'].append(Tracet[i1])
                GreaterThanCentre = True
    #c = np.empty(len(m[:,0])); c.fill(megno)
    points0['points']=np.array(points0['points'])
    points1['points']=np.array(points1['points'])
    points0['t']=np.array(points0['t'])
    points1['t']=np.array(points1['t'])
    return points0, points1

def NearestFinder(array,value):
    idx = np.argmin(np.abs(array-value))
    return idx
    
def DistanceFinder(Data):
    try:
        ADSA, dt, NumberOfSteps, Arg0 , Arg1 ,Index0,Index1,phase=Data
        ADSA, Traces=SimulationLoop( ADSA, dt, NumberOfSteps, Arg0 , Arg1 ,phase,Index0,Index1)
        points0,points1 = CrossAnalysis(Traces[1][:,0],Traces[4][:,0],Traces[1],Traces[0])
        Distance = np.zeros([1])
        DistanceAv = np.zeros([1])
        DistanceAcc = np.zeros([1])
        DistanceAccAv = np.zeros([1])
        print(points0)
        if len(points0['t'] )>Traces[-1]/30000/3:
            Distance0=np.array(points0['points'][1:,:])-np.array(points0['points'][0:-1,:])  
            
            Distance0=np.vstack((np.zeros(Distance0[0].shape), Distance0) )
#            Distance1=np.array(points1['points'][1:,:])-np.array(points1['points'][0:-1,:])    
#            Distance1=np.append(np.zeros(Distance1[0].shape), Distance1 )
            print('Distance0'+str(Distance0))
            Distance=np.linalg.norm(Distance0,axis=1)
            DistanceAv = np.average(Distance)
            DistanceAcc=Distance[1:]-Distance[0:-1]    
            DistanceAcc=np.append(np.zeros(DistanceAcc[0].shape), DistanceAcc) 
            DistanceAccAv=np.average(DistanceAcc)
    except:
        traceback.print_exc(file=sys.stderr)
    return Index0, Index1, ADSA, Arg0 , Arg1, Distance, DistanceAv, DistanceAcc, DistanceAccAv 
    
#def ParameterOptimizer(AMBONs,gm , TDm):
def DataGenerator(ADSA,dt, NumberOfSteps, Arg0 , Arg1 , phase):
    Index0=0
    Index1=0
    while Index0<len(Arg0)and Index1<len(Arg1[0]):
        data=[ADSA,dt, NumberOfSteps,Arg0[Index0,Index1], Arg1[Index0,Index1],Index0,Index1,phase]
        yield data
        if Index1<len(Arg0[0])-1:
            Index1 +=1
        else:    
            #if Index0<len(gm)-1:
            Index1=0            
            Index0 +=1
            
def SimulationLoop(ADSA,dt, NumberOfSteps, Arg0 , Arg1 , phase=0,Index0=0,Index1=0):    
    ADSA.tauWV = Arg0
    ADSA.aWV = Arg1
    
    ADSA.InitRecording(NumberOfSteps)  
    Tracet = np.zeros(NumberOfSteps)           
    for step in range(NumberOfSteps):
    #        WeightersLast = copy.deepcopy(Weighters)
    #        WeighterVarRatesLast = copy.deepcopy(WeighterVarRates)
        ADSA.StateUpdate()
        ADSA.StepSynapseDynamics( SimulationTimeInterval,0)
        if  ADSA.RecordingState:
            ADSA.Recording()  
            Tracet[step] = step*SimulationTimeInterval
            #%%
        if step%(100000./dt)<1:
           print ('phase=%s,Index0=%d, Index1=%d, tauWV=%s, aWV=%s, step=%s'%(phase,Index0,Index1,ADSA.tauWV, ADSA.aWV,step))  
#    Tracet, TraceWeighters, TraceWeighterVarRates, TraceWeighterInDendrite, traceWeightersCentre,traceWeighterVarDamping 
    NeuonNumber=1
    newSlice= [slice(None)]*3
    newSlice[1]=NeuonNumber
    Traces = Tracet, ADSA.Trace['Weighters'][newSlice], ADSA.Trace['WeighterVarRates'][newSlice], ADSA.Trace['WeighterInDendrite'][newSlice], ADSA.Trace['WeightersCentre'][newSlice], ADSA.Trace['WeighterVarDamping'][newSlice], ADSA.Trace['EquivalentVolume'][newSlice]
  

    return ADSA, Traces            
            
if __name__=="__main__":
    InintialDS =1
    SearchParameters =0
    InintialSearch =1
    SingleSimulation = 1
    PlotPhasePotrait = 0
    TimOfRecording=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    if InintialDS:
        NumberOfNeuron=2
        NumberOfSynapses = 45# N =3 tauWV =50; #N = 6 tauWV = 25
        Weighters= np.ones((NumberOfNeuron,NumberOfSynapses))*0.2+ 0.1 * np.random.rand(NumberOfNeuron,NumberOfSynapses)#np.random.rand(NumberOfNeuron,NumberOfSynapses) #0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)  #
        WeighteAmountPerSynapse = 1
        WeighterInDendrite = WeighteAmountPerSynapse* NumberOfSynapses - Weighters.sum(axis=1)
        WeighterVarRates = np.zeros((NumberOfNeuron,NumberOfSynapses))
    
        
        
    #    TraceWeighters = np.zeros((NumberOfSteps,NumberOfNeuron,NumberOfSynapses))
    #    TraceWeighterVarRates  = np.zeros((NumberOfSteps,NumberOfNeuron,NumberOfSynapses))
    #    TraceWeighterInDendrite = np.zeros((NumberOfSteps,NumberOfNeuron))
#        CW = 100
#        tauWV =60#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#130   100
#        rWV = 7000 #7000
#        scale=1
#        damping =2
#        CW = 100
#        SimulationTimeInterval = 30
#        CW = 100

## ratio of intergration of postive value oscillation and nagative value oscillation is low  *** when receptor amount 1
#        tauWV =500#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#130   100
#        rWV = 7000 #7000
#        scale=1
#        damping =2
#        CW = 100
#        SimulationTimeInterval = 10

## oscillation with periods of 300 to 500 seconds *** when receptor amount 1
#
#        tauWV =0.02#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 17#170#130   100
#        rWV = 7000*500*10 #7000
#        scale=1
#        damping =2*7000
#        CW = 100
#        SimulationTimeInterval = 100

#TODO # oscillation with periods of 300 to 500 seconds  *** when receptor amount 10
#
#        tauWV =0.02#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 34#170#130   100
#        rWV = 7000*500*200#7000
#        scale=1
#        damping =7000
#        CW = 100
#        SimulationTimeInterval = 100
        
        
### oscillation with periods of 500seconds  *** when receptor amount 1
#
#        tauWV =0.1#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 34#170#130   100
#        rWV = 7000*500*5#7000
#        scale=1
#        damping =2*7000
#        CW = 100
#        SimulationTimeInterval = 100

## oscillation with periods of 50seconds  *** when receptor amount 10
#
#        tauWV =0.1#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100#170#130   100
#        rWV = 7000*500*3000#7000
#        scale=1
#        damping =2*7000
#        CW = 100
#        SimulationTimeInterval = 100
#        
# oscillation with periods of 17 seconds  *** when receptor amount 1

        tauWV =0.5#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
        aWV = 100#170#130   100
        rWV = 7000*500#7000
        scale=1
        damping =2*7000
        CW = 100
        SimulationTimeInterval = 20

## oscillation with periods of 50seconds  *** when receptor amount 1
#
#        tauWV =0.7#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#170#130   100
#        rWV = 7000*500#7000
#        scale=1
#        damping =2*7000
#        CW = 100
#        SimulationTimeInterval = 20
#        
        
        #        tauWV =500#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 85#130   100
#        rWV = 3500 #7000
#        scale=1
#        damping =2
#        CW = 100
#        SimulationTimeInterval = 10


#        tauWV = 20#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 130#100
#        rWV = 5000
#        scale=1
#        damping =2
#        SimulationTimeInterval = 30
#        CW = 100
#        tauWV = 40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100
#        rWV = 10000
#        scale=1
#        damping =2
#        SimulationTimeInterval = 30
    #    CW = 100
    #    tauWV = 40#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
    #    aWV = 0.0005#0.02 #100 
    #    rWV = 200000
    #    scale=1
 #0.4 0.05     
#        WeightersCentre = np.ones((NumberOfNeuron,NumberOfSynapses))*0.2+ 0.04*np.mgrid[0:NumberOfNeuron,0:NumberOfSynapses][1]#(NumberOfSynapses) #* np.random.rand(NumberOfNeuron,NumberOfSynapses) # 0.6 * np.random.rand(NumberOfNeuron,NumberOfSynapses) #np.array([4, 1, 1, 1, 1])#np.ones(NumberOfSynapses)*0.4 + 0.3 * np.random.rand(NumberOfSynapses) #np.ones(NumberOfSynapses)/2 + [0.8, 0.1,0.1, 0.1, 0]  #[0.2, 0.1, 0]
  
        WeightersCentre = np.ones((NumberOfNeuron,NumberOfSynapses))*0.3+0.02*np.mgrid[0:NumberOfNeuron,0:NumberOfSynapses][1]#(NumberOfSynapses) #* np.random.rand(NumberOfNeuron,NumberOfSynapses) # 0.6 * np.random.rand(NumberOfNeuron,NumberOfSynapses) #np.array([4, 1, 1, 1, 1])#np.ones(NumberOfSynapses)*0.4 + 0.3 * np.random.rand(NumberOfSynapses) #np.ones(NumberOfSynapses)/2 + [0.8, 0.1,0.1, 0.1, 0]  #[0.2, 0.1, 0]
        WeighterVarDamping = np.ones((NumberOfNeuron,NumberOfSynapses)) * damping #np.array([10, 2, 2, 2, 2]) #[10,2,2,2,2]       #[2,2,2]
#        WeighterVarDamping[0,1] = 4
        Parameters = [CW, tauWV, aWV , rWV, WeightersCentre,WeighterVarDamping]
        
        ADSA=DynamicSynapseArray( NumberOfSynapses = [NumberOfNeuron, NumberOfSynapses], CW = CW, tauWV = tauWV, aWV = aWV, rWV = rWV,scale=scale, \
                    WeightersCentre = WeightersCentre , WeighterVarDamping = WeighterVarDamping, WeighteAmountPerSynapse = WeighteAmountPerSynapse, \
                    Weighters = Weighters, WeighterVarRates = WeighterVarRates,WeightersCentreUpdateCompensate = 0)
#        ADSA.DampingUpdateRate=0
        SimulationTimeLenth = 60*60*1000
        
        dt = SimulationTimeInterval 
        NumberOfSteps = int(SimulationTimeLenth/SimulationTimeInterval)
        Tracet = np.zeros(NumberOfSteps)
        ADSA.InitRecording(NumberOfSteps) 
 #       ADSA.InitRecording(NumberOfSteps)  

    if SearchParameters :
        if InintialSearch:
            searchSamples=[10,10]
            centreSearchSamples=np.floor(np.array(searchSamples)/2).astype(int)
            DistanceAvLastTime=0
            traceDistanceAccAv=np.zeros(searchSamples)
            traceDistanceAv=np.zeros(searchSamples)
            Arg0Indexs=[]
            Arg0Maxs=[]
            Arg1Maxs=[]
            Scales=[]
            phase=1
            ax1=[None for i in range(phase)]
            fig=[None for i in range(phase)]
            img=[None for i in range(phase)]
            DistanceAccAvThisTime=0
            numberOfProcess=15
            DistanceAccAv=np.zeros(searchSamples)
            DistanceMax=np.zeros(searchSamples)
            pool = Pool(processes=numberOfProcess)
            #_unordered
            Arg0spaceLim=[10,40]
            #gspaceLim=[0.7,0.72]
            Arg0Max=np.average(Arg0spaceLim)
            Arg1spaceLim=[100,200]
            #Arg1spaceLim=[38, 38.6]
            Arg1Max=np.average(Arg1spaceLim)
            Arg0space=np.linspace(Arg0spaceLim[0],Arg0spaceLim[1], num=searchSamples[0])
            Arg1space=np.linspace(Arg1spaceLim[0],Arg1spaceLim[1], num=searchSamples[1])
            Scale=np.array([Arg0spaceLim[1]-Arg0spaceLim[0],Arg1spaceLim[1]-Arg1spaceLim[0]])
        for i1 in range(phase):
            
            Arg0space=np.linspace(Arg0Max-Scale[0]/2,Arg0Max+Scale[0]/2, num=searchSamples[0])
            Arg1space=np.linspace(Arg1Max-Scale[1]/2,Arg1Max+Scale[1]/2, num=searchSamples[1]) 
            randSearchRate0=(np.random.random_sample(Arg0space.shape)-0.5)*0#.1
            randSearchRate0[centreSearchSamples[0]]=0
            randSearchRate1=(np.random.random_sample(Arg1space.shape)-0.5)*0#.1
            randSearchRate0[centreSearchSamples[1]]=0
            Arg0m , Arg1m  = np.meshgrid(Arg0space*(1+randSearchRate0), Arg1space*(1+randSearchRate1)) 
            print (Arg0m, Arg1m)
    #        traceV,traceU,traceI,Iss=SimulationLoop(AMBONs,gm , Arg1m )
            iTestResultsOfTests= pool.imap_unordered(DistanceFinder, DataGenerator(ADSA, dt, NumberOfSteps, Arg0m , Arg1m, i1))  #
            for AResult in iTestResultsOfTests:
                Index0, Index1, ADSA, Arg0 , Arg1, Distance, DistanceAv, DistanceAcc, DistanceAccAv  =AResult
                traceDistanceAv[Index0, Index1]=DistanceAv
                traceDistanceAccAv[Index0, Index1]=DistanceAccAv
#                traceDistanceAccAv.append(DistanceAccAv)
#                traceDistanceAv.append(DistanceAv)
                print('Distance:'+str(Distance))
#            MaxIndex=np.unravel_index(np.argmax(traceDistanceAv),traceDistanceAv.shape)
            MaxIndex=np.unravel_index(np.argmax(traceDistanceAv),traceDistanceAccAv.shape)
#            Arg0Index=int(MaxIndex % searchSamples[0])
            Arg0Max=Arg0space[MaxIndex[0]]
#            Arg0Indexs.append(Arg0Index)
#            Arg0Maxs.append(Arg0Max)
#            Arg1Index=np.floor(MaxIndex/searchSamples[0]).astype(int)
            Arg1Max=Arg1space[MaxIndex[1]]
#            Arg1Maxs.append(Arg1Max)


            
            DistanceAvThisTime=traceDistanceAv[MaxIndex]
            Improve=DistanceAvThisTime-DistanceAvLastTime
            if Improve <-0.1:
                break
                print("negative imporve")
            else:
                DistanceAvLastTime=DistanceAvThisTime
                Scale=Scale*0.5#/np.average(searchSamples)*2
                Scales.append(Scale)
#            for i1 in range(10):
#                print ("Arg0Max=%64f, Arg1Max=%64f, i=%s,DistanceAccAvThisTime=%64f, Improve=%64f"%(Arg0Max,Arg1Max, i1,DistanceAvThisTime,Improve))
        #plt.imshow(traceDistanceAccAv)
            print ("Arg0Max=%f, Arg1Max=%f, i=%s,DistanceAccAvThisTime=%f, Improve=%f"%(Arg0Max,Arg1Max, i1,DistanceAvThisTime,Improve))
            #%%
            fig[i1]=plt.figure()
            
            x,y=np.mgrid[slice(Arg0space[0],Arg0space[-1],searchSamples[0]*1j),slice(Arg1space[-1],Arg1space[0],searchSamples[1]*1j)]
            ax1[i1] = fig[i1].add_subplot(111)
            img[i1]=ax1[i1].pcolormesh(x,y,traceDistanceAv)
            fig[i1].colorbar(img[i1],ax=ax1[i])
            fig[i1].show()
            #%%
        print (traceDistanceAccAv,traceDistanceAv)
        ADSA.tauWV = Arg0Max
        ADSA.aWV = Arg1Max
#%%       
    if SingleSimulation: 
#%%     
#        DampingUpdateRateCache = ADSA.DampingUpdateRate          
        for step in range(NumberOfSteps):
        #        WeightersLast = copy.deepcopy(Weighters)
        #        WeighterVarRatesLast = copy.deepcopy(WeighterVarRates)
#            if step * SimulationTimeInterval<60*60*1000:
#                ADSA.DampingUpdateRate=0
#            else:
#                ADSA.DampingUpdateRate= DampingUpdateRateCache
            ADSA.StateUpdate()

            ADSA.StepSynapseDynamics( SimulationTimeInterval,0, Compensate='Dynamic0')
            #print (ADSA.EquivalentVolume)
#            ADSA.StepSynapseDynamics( SimulationTimeInterval,np.random.rand(1)/10000, Compensate='Dynamic')

            if  ADSA.RecordingState:
                ADSA.Recording(N_Append=1)  
                Tracet[step] = step*SimulationTimeInterval
                #%
            if step % 1000 == 0:
                print('%d of %d steps'%(step,NumberOfSteps))
        
    #    Tracet, TraceWeighters, TraceWeighterVarRates, TraceWeighterInDendrite, traceWeightersCentre,traceWeighterVarDamping 
        ADSA.RecordingFinish()
        NeuonNumber=0
        newSlice= [slice(None)]*3
        newSlice[1]=NeuonNumber
        Traces = Tracet, ADSA.Trace['Weighters'][newSlice], ADSA.Trace['WeighterVarRates'][newSlice], ADSA.Trace['WeighterInDendriteConcentration'][newSlice], ADSA.Trace['WeightersCentre'][newSlice], ADSA.Trace['WeighterVarDamping'][newSlice], ADSA.Trace['EquivalentVolume'][newSlice]
#%%
        FigureDict,ax = plot(TimOfRecording, Traces, path='', savePlots=True, linewidth= 0.2) #path=
        
        #%%
    #    for angle in range(0, 360):
    #        ax.view_init(30, angle)
    #        #plt.draw()
    #        plt.pause(.0001)
    #%%
        UpHalfWeightSum= (ADSA.Trace['Weighters'][newSlice]-WeightersCentre[NeuonNumber,:])[ADSA.Trace['Weighters'][newSlice]>WeightersCentre[NeuonNumber,:]].sum()
        UpHalfTime=ADSA.Trace['Weighters'][newSlice][ADSA.Trace['Weighters'][newSlice]>WeightersCentre[NeuonNumber,:]].shape[0]*dt
        DownHalfWeightSum= (WeightersCentre[NeuonNumber,:]-ADSA.Trace['Weighters'][newSlice])[ADSA.Trace['Weighters'][newSlice]<WeightersCentre[NeuonNumber,:]].sum()
        DownHalfTime=ADSA.Trace['Weighters'][newSlice][ADSA.Trace['Weighters'][newSlice]<WeightersCentre[NeuonNumber,:]].shape[0]*dt
        print("UpHalfWeightSum:  %f, UpHalfTime:  %f"%(UpHalfWeightSum, UpHalfTime))
        print("DownHalfWeightSum:%f, DownHalfTime:%f"%(DownHalfWeightSum, DownHalfTime))
        print("UDWeightRate:%f, UDTimeRate:%f"%(UpHalfWeightSum/DownHalfWeightSum, float(UpHalfTime)/DownHalfTime))
# %%
    if PlotPhasePotrait :
        PhasePotraitfig, PhasePotraitax = ADSA.PlotPhasePortrait( xlim=[0,1], ylim=[-0.00002, 0.00002], fig=None, ax=None, inputs=[1, 0.5], Parameters=None)
        PhasePotraitax.plot(ADSA.Trace['Weighters'],ADSA.Trace['WeighterVarRates'])