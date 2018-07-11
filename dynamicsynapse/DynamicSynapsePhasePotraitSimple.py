#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:05:14 2017

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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def rk4(h, y, inputs, Parameters, f):
        k1 = f(y, inputs, Parameters)
#        print(y)
#        print(h)
#        print(k1)
#        print(y + 0.5*h*k1)
        k2 = f(y + 0.5*h*k1, inputs, Parameters)
        k3 = f(y + 0.5*h*k2, inputs, Parameters)
        k4 = f(y + k3*h, inputs, Parameters)
        return y + (k1 + 2*(k2 + k3) + k4)*h/6.0
        
class DynamicSynapseArray:
    def __init__(self, NumberOfSynapses = 5, CW = 100, tauWV = 40, aWV = 100, rWV = 5000, scale=10,  \
                WeightersCentral = None , WeighterVarDamping = None, WeighteAmountPerSynapse = 1, \
                Weighters = None, WeighterVarRates = None, WeightersCentralUpdateRate = 0.000012,\
                DampingUpdateRate = 0.0000003/100 , WeightersCentralUpdateCompensate =0, MaxDamping = 10):
        #self.NumberOfNeuron=NumberOfSynapses[0]
        self.NumberOfSynapses = NumberOfSynapses#[1]
        self.CW = CW
        self.tauWV = tauWV
        self.aWV = aWV 
        self.rWV = rWV
        self.scale =scale
        
        self.WeightersCentral = WeightersCentral if WeightersCentral is not None else np.ones(NumberOfSynapses)/2+[[0.1, 0.1,0.1, 0.1, 0.1]]*NumberOfSynapses[0] #[0.2, 0.1, 0]
        self.WeighterVarDamping = WeighterVarDamping  if WeighterVarDamping is not None else  np.ones(NumberOfSynapses) * [[2,2,2,2,2]]*NumberOfSynapses[0] #[2,2,2]
        self.DampingUpdateRate = DampingUpdateRate
        self.GetParameters = lambda: [self.CW, self.tauWV, self.aWV , self.rWV, self.scale, self.WeightersCentral,self.WeighterVarDamping]
        self.Parameters = [self.CW, self.tauWV, self.aWV , self.rWV, self.scale, self.WeightersCentral,self.WeighterVarDamping]

        self.WeighteAmountPerSynapse = WeighteAmountPerSynapse
        
        
        self.WeightersLast = Weighters if Weighters is not None else 0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)
        self.Weighters = self.WeightersLast
        self.WeighterInAxon =  self.WeighteAmountPerSynapse* self.NumberOfSynapses[1] - self.WeightersLast
        self.WeighterInAxonConcentration = self.WeighterInAxon/self.NumberOfSynapses[1]
        self.WeighterVarRatesLast = WeighterVarRates if WeighterVarRates is not None else np.zeros(NumberOfSynapses)
        self.WeighterVarRates = self.WeighterVarRatesLast
        
        self.EquivalentVolume = (1+(2*self.WeightersCentral-(self.WeighterInAxonConcentration+self.WeightersLast))/((self.WeighterInAxonConcentration+self.WeightersLast)-self.WeightersCentral))

        self.WeightersCentralUpdateRate = WeightersCentralUpdateRate
        self.WeightersCentralUpdateCompensate = WeightersCentralUpdateCompensate
        
        self.MaxDamping = MaxDamping
        
    def Derivative (self, state=None , inputs=None, Parameters=None):
        if state is not None:
            WeightersLast, WeighterVarRatesLast = state
        else:
            WeightersLast, WeighterVarRatesLast = self.WeightersLast, self.WeighterVarRatesLast 
        if inputs is not None:
            WeighterInAxonConcentration=inputs
        else:
            WeighterInAxonConcentration=self.WeighterInAxonConcentration
        if Parameters is not None:
            CW, tauWV, aWV, rWV, scale, WeightersCentral,WeighterVarDamping = Parameters
        else:
            CW, tauWV, aWV, rWV, scale,  WeightersCentral,WeighterVarDamping = self.Parameters()
        #print(WeighterVarRatesLast , WeighterInAxonConcentration , WeightersLast , self.scale)
        EquivalentVolume = (1+(2*WeightersCentral-(WeighterInAxonConcentration+WeightersLast))/((WeighterInAxonConcentration+WeightersLast)-WeightersCentral))
        self.EquivalentVolume = EquivalentVolume
#        DWeighters =  WeighterVarRatesLast * WeighterInAxonConcentration * WeightersLast /self.scale*2
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
#        DWeighters =  WeighterVarRatesLast  * ( WeighterInAxonConcentration + WeightersLast/EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInAxonConcentration - WeightersLast/self.EquivalentVolume)) /2  /self.scale    
#        DWeighterVarRates = (  (WeighterInAxonConcentration - WeightersLast/EquivalentVolume  \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /scale 

        DWeighters =  WeighterVarRatesLast  * ( WeighterInAxonConcentration + WeightersLast/EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInAxonConcentration - WeightersLast/EquivalentVolume)) /2  /self.scale    
        DWeighterVarRates = (  (tauWV*(WeighterInAxonConcentration - (WeightersLast/EquivalentVolume)**2 ) \
        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5)) - WeighterVarDamping*WeighterVarRatesLast )  /scale / rWV
#        print('DWeighters, DWeighterVarRates')
#        print(DWeighters, DWeighterVarRates)
##        DWeighters =  WeighterVarRatesLast * WeighterInAxonConcentration * WeightersLast /self.scale*2
##        DWeighters =  WeighterVarRatesLast /self.scale/2
##        CW = 100
##        tauWV = 17.8#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
##        aWV = 100 
##        rWV = 5000
##        scale=1
#        DWeighters =  WeighterVarRatesLast  * ( WeighterInAxonConcentration + WeightersLast/self.EquivalentVolume +np.sign(WeighterVarRatesLast)*(WeighterInAxonConcentration - WeightersLast/self.EquivalentVolume)) /2  /self.scale    
#        DWeighterVarRates = (  (WeighterInAxonConcentration - WeightersLast/self.EquivalentVolume  \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /self.scale 

#very chaos, no distinguish between pump rate and transport speed
#        CW = 100
#        tauWV = 17.8--50#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100 
#        rWV = 5000
#        scale=1
#        DWeighters =  WeighterVarRatesLast /self.scale
#        DWeighterVarRates = (  (WeighterInAxonConcentration - WeightersLast/self.EquivalentVolume  \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV * ( WeighterInAxonConcentration + WeightersLast +np.sign(WeighterVarRatesLast)*(WeighterInAxonConcentration - WeightersLast)) /2  /self.scale
#instantious catch-up lateral mobility resistance, no pump resistance
##        CW = 100
##        tauWV = 40#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
##        aWV = 0.0005#0.02 #100 
##        rWV = 200000
##        scale=1
#        DWeighters =  WeighterVarRatesLast +  (WeighterInAxonConcentration - WeightersLast/self.EquivalentVolume)/ rWV /self.scale
#        DWeighterVarRates = (  \
#        (aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5) - WeighterVarDamping*WeighterVarRatesLast )  * ( WeighterInAxonConcentration + WeightersLast +np.sign(WeighterVarRatesLast)*(WeighterInAxonConcentration - WeightersLast)) /2 )/self.scale

#original
#        DWeighters =  WeighterVarRatesLast * WeighterInAxonConcentration * WeightersLast /self.scale
#        DWeighterVarRates = (  (WeighterInAxonConcentration - WeightersLast/(1+(2*WeightersCentral-(WeighterInAxonConcentration+WeightersLast))/((WeighterInAxonConcentration+WeightersLast)-WeightersCentral) ) \
#        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV  - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /self.scale
        return np.array([DWeighters, DWeighterVarRates])
    
    #def SynapseDerivative(WeightersLast, WeighterVarRatesLast, WeighterInAxon, Parameters):
    #    tauW, tauWV, aWV = Parameters
    #    DWeightersLast = WeighterVarRatesLast
    #    DWeighterVarRatesLast = aWV * (  WeighterInAxon - WeightersLast  + np.sign(WeighterVarRatesLast)*np.power(WeighterVarRatesLast, 0.5))
    #    return DWeightersLast, DWeighterVarRatesLast
    def Jacobian(self, state=None , inputs=None, Parameters=None):
        if state is not None:
            WeightersLast, WeighterVarRatesLast = state
        else:
            WeightersLast, WeighterVarRatesLast = self.WeightersLast, self.WeighterVarRatesLast 
        if inputs is not None:
            WeighterInAxonConcentration=inputs
        else:
            WeighterInAxonConcentration=self.WeighterInAxonConcentration
        if Parameters is not None:
            CW, tauWV, aWV, rWV, scale, WeightersCentral,WeighterVarDamping = Parameters
        else:
            CW, tauWV, aWV, rWV, scale, WeightersCentral,WeighterVarDamping = self.Parameters()
        DDWDW =  WeighterVarRatesLast * ( 1 -np.sign(WeighterVarRatesLast) ) /2 /self.scale
        DDWDWV =  ( WeighterInAxonConcentration + WeightersLast +np.sign(WeighterVarRatesLast)*(WeighterInAxonConcentration - WeightersLast)) /2 /self.scale
        DDWVDW = (  (WeighterInAxonConcentration - WeightersLast/(1+(2*WeightersCentral-(WeighterInAxonConcentration+WeightersLast))/((WeighterInAxonConcentration+WeightersLast)-WeightersCentral) ) \
        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /self.scale
        DDWVDDWV = (  (WeighterInAxonConcentration - WeightersLast/(1+(2*WeightersCentral-(WeighterInAxonConcentration+WeightersLast))/((WeighterInAxonConcentration+WeightersLast)-WeightersCentral) ) \
        +aWV * np.sign(WeighterVarRatesLast)*np.power(np.abs(WeighterVarRatesLast), 0.5))/ rWV - WeighterVarDamping*WeighterVarRatesLast ) /  tauWV  /self.scale
        return np.array([DWeighters, DWeighterVarRates])


    def StepSynapseDynamics(self, dt, ModulatorAmount):
#        ModulatorAmount=np.array(ModulatorAmount)
#        ModulatorAmount=ModulatorAmount.reshape(np.append(ModulatorAmount.shape,1).astype(int))
        self.Parameters = [self.CW, self.tauWV, self.aWV , self.rWV, self.scale, self.WeightersCentral,self.WeighterVarDamping]
        self.Weighters, self.WeighterVarRates = rk4(dt, np.array([self.WeightersLast, self.WeighterVarRatesLast]), self.WeighterInAxonConcentration, self.Parameters, self.Derivative)
#        print('self.Weighters')
#        print(self.Weighters)
#        if np.isnan(self.Weighters):
#            pass
        self.WeighterInAxon =  self.WeighteAmountPerSynapse* self.NumberOfSynapses[1] - self.WeightersLast
        #print(self.WeighterInAxon, self.WeighteAmountPerSynapse, self.NumberOfSynapses[1] , self.WeightersLast.sum(axis=1,keepdims=True))
        self.WeighterInAxonConcentration = self.WeighterInAxon/self.NumberOfSynapses[1]
        self.WeightersCentral += (self.Weighters-self.WeightersCentral)*ModulatorAmount *self.WeightersCentralUpdateRate*dt*(1+self.WeightersCentralUpdateCompensate*(self.Weighters>self.WeightersCentral)) #0.000015##0.00002
#        self.WeightersCentral += (self.Weighters-self.WeightersCentral)*ModulatorAmount *self.WeightersCentralUpdateRate*dt #0.000015##0.00002
        #print(self.WeightersCentral)
        
        self.WeighterVarDamping += (self.MaxDamping-self.WeighterVarDamping)*self.WeighterVarDamping*ModulatorAmount*self.DampingUpdateRate *dt #
        #print(self.WeighterVarDamping)
        return self.Weighters, self.WeighterVarRates, self.WeighterInAxon, self.WeighterInAxonConcentration
        
    def StateUpdate(self):
        self.WeightersLast, self.WeighterVarRatesLast = self.Weighters, self.WeighterVarRates 
        
    def InitRecording(self, lenth, SampleStep = 1):
        self.RecordingState = True
        self.RecordingLenth = lenth
        self.RecordingInPointer = 0
        self.Trace = {'Weighters':np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'WeightersCentral' : np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'WeighterVarRates' : np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()), \
                        'WeighterVarDamping' : np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()),\
                        'WeighterInAxonConcentration' :  np.zeros(np.append(np.append(lenth,self.NumberOfSynapses[0]),1).ravel()), \
                        'EquivalentVolume':np.zeros(np.append(lenth,self.NumberOfSynapses).ravel()) \
                        }
    def Recording(self):
        Temp = None
        for key in self.Trace:
            exec("Temp = self.%s" % (key))
#            print ("Temp = self.%s" % (key))
#            print(Temp)
            self.Trace[key][self.RecordingInPointer, :] = Temp
        self.RecordingInPointer += 1
        if self.RecordingInPointer>= self.RecordingLenth:
            self.RecordingInPointer = 0
                
#%%                
    def PlotPhasePortrait(self, xlim, ylim, fig=None, ax=None,  inputs=None, Parameters=None):
#        fig2 = plt.figure(figsize=(10, 20))
#        ax4 = fig2.add_subplot(1,1,1)
        if fig == None or ax == None:
            fig, ax = plt.subplots(1,sharex=False , figsize=(20, 12))#

        if Parameters is not None:
            CW, tauWV, aWV, rWV, scale, WeightersCentral,WeighterVarDamping = Parameters
        else:
            CW, tauWV, aWV, rWV, scale, WeightersCentral,WeighterVarDamping = self.GetParameters()
        if inputs is not None:
            d, h  = WeighterInAxonConcentration, WeightersCentral=inputs
        else:
            d = WeighterInAxonConcentration=self.WeighterInAxonConcentration
        a= aWV / tauWV
        b= WeighterVarDamping / tauWV
        h = WeightersCentral
        s=1
        Wis=np.linspace(xlim[0],xlim[1],num=10000)
        w= Wis
        vis = np.linspace(ylim[0],ylim[1],num=10000)
        d=self.WeighteAmountPerSynapse-Wis
        colors=['r','c']
#        temp1 = 2 * (b*h*s)**2
#        temp2 = a * (h*s)**(3/2)
#        temp3 = a**2 * h * s
#        temp4 = 4 * (b*d*h*s - b*d*s*w + b*h*w - b*w**2)
#        temp5 = (a*h*s)**2 
#        temp6 = 2*b*d*(h*s)**2 - 2*b*d*h*s**2*w +2*b*h**2*s*w - 2*b*h*s*w**2
#        ax.plot( Wis, (-temp2 * np.sqrt(temp3+temp4) + temp5 + temp6)/ temp1, lw=2, label='v-nullcline 0' ) 
#        ax.plot( Wis, (temp2 * np.sqrt(temp3+temp4) + temp5 + temp6)/ temp1, lw=2, label='v-nullcline 1' ) 
#        ax.plot( Wis, (-temp2 * np.sqrt(temp3-temp4) - temp5 + temp6)/ temp1, lw=2, label='v-nullcline 2' ) 
#        ax.plot( Wis, (temp2 * np.sqrt(temp3-temp4) - temp5 + temp6)/ temp1, lw=2, label='v-nullcline 3' ) 
        
#        temp1 = 2 * b**2
#        temp2 = a 
#        temp3 = a**2
#        temp4 = 4 * (b*d - b*w)
#        temp5 = a**2 
#        temp6 = 2*b*d - 2*b*w
#        ax.plot( Wis, (-temp2 * np.sqrt(temp3+temp4) + temp5 + temp6)/ temp1, lw=2, label='v-nullcline 0' ) 
#        ax.plot( Wis, (temp2 * np.sqrt(temp3+temp4) + temp5 + temp6)/ temp1, lw=2, label='v-nullcline 1' ) 
#        ax.plot( Wis, (-temp2 * np.sqrt(temp3-temp4) - temp5 + temp6)/ temp1, lw=2, label='v-nullcline 2' ) 
#        ax.plot( Wis, (temp2 * np.sqrt(temp3-temp4) - temp5 + temp6)/ temp1, lw=2, label='v-nullcline 3' )   
#        ax.plot( Wis,np.zeros(Wis.shape) , lw=2, label='w-nullcline' ) 
#        ax.plot(a*np.sign(vis)*np.sqrt(np.abs(vis))-b*vis+d, vis, lw=2, label='v-nullcline 4' )
        
        temp1 = 2 * b**2
        temp2 = a 
        temp3 = a**2
        temp4 = 4 * (b*d - b*w**2)
        temp5 = a**2 
        temp6 = 2*b*d - 2*b*w**2
        ax.plot( Wis, (-temp2 * np.sqrt(temp3+temp4) + temp5 + temp6)/ temp1, lw=2, label='v-nullcline 0' ) 
        ax.plot( Wis, (temp2 * np.sqrt(temp3+temp4) + temp5 + temp6)/ temp1, lw=2, label='v-nullcline 1' ) 
        ax.plot( Wis, (-temp2 * np.sqrt(temp3-temp4) - temp5 + temp6)/ temp1, lw=2, label='v-nullcline 2' ) 
        ax.plot( Wis, (temp2 * np.sqrt(temp3-temp4) - temp5 + temp6)/ temp1, lw=2, label='v-nullcline 3' )        
        
        ax.plot( Wis,np.zeros(Wis.shape) , lw=2, label='w-nullcline' ) 
        ax.plot(np.sqrt(a*np.sign(vis)*np.sqrt(np.abs(vis))-b*vis+d), vis, lw=2, label='v-nullcline 4' )
        
        ax.set_ylim (ylim)
#        ax.axvline( 0 , lw=2, label='w-nullcline' ) #
        Wispace=np.linspace(xlim[0],xlim[1], num=30)
        Vspace=np.linspace(ylim[0],ylim[1], num=20)
        Wistep=Wispace[1]-Wispace[0]
        Vstep=Vspace[1]-Vspace[0]

        W1 , V1  = np.meshgrid(Wispace, Vspace) 
        print(W1)
        print(V1)
        DW1, DV1=self.Derivative(state=[W1, V1] , inputs=self.WeighteAmountPerSynapse-W1, Parameters=[CW, tauWV, aWV, rWV, scale, WeightersCentral,WeighterVarDamping]  )
#        VectorZ=DW1+ DV1*1j
#        M = np.log(np.hypot(DW1, DV1))
        M = np.greater(DV1, 0)
        ax.quiver(W1 , V1, DW1, DV1, (M), width=0.002, angles='xy')#pivot='mid')



        #ax.legend(bbox_to_anchor=(0.6, 0.2), loc=2, borderaxespad=0.,prop={'size':12})
        ax.legend(prop={'size':12})
        ax.grid()
        return [fig, ax]
            
def plot(TimOfRecording, Traces = None, path='', savePlots = False, StartTimeRate=0.3, DownSampleRate=10,linewidth =1, FullScale=False):
#    plt.rc('axes', prop_cycle=(cycler('color',['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','k'])))
    mpl.rcParams['axes.prop_cycle']=cycler('color',['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','b','k'])
#    mpl.rcParams['axes.prop_cycle']=cycler(color='category20')
    if Traces is not None:
        Tracet, TraceWeighters, TraceWeighterVarRates, TraceWeighterInAxonConcentration, TraceWeightersCentral,TraceWeighterVarDamping,TraceEquivalentVolume = Traces
#        else:
#            for key in self.Trace:
#                exec("Trace%s = self.Trace[%s]" % (key,key))
    TracetInS=Tracet.astype(float)/1000
    NumberOfSteps = len(TracetInS)
    if StartTimeRate == 0:
          StartStep = 0
    else:
          StartStep = NumberOfSteps - int(NumberOfSteps*StartTimeRate)
          
    figure1 = plt.figure()  
    labels = [str(i) for i in range(TraceWeighters.shape[1])]
    figure1lines = plt.plot(TracetInS, TraceWeighters,  label=labels, linewidth= linewidth)
    plt.legend(figure1lines, labels)
    plt.xlabel('Time (s)')
    plt.title('Instantaneous Synaptic Strength')
    
    figure2 = plt.figure();
    plt.plot(TracetInS, TraceWeighterVarRates,linewidth= linewidth)
    plt.xlabel('Time (s)')
    plt.title("'Pump' rate")
    
    figure3 = plt.figure()
    ConcentrationLines =plt.plot(TracetInS[::DownSampleRate], TraceWeighters[::DownSampleRate]/TraceEquivalentVolume[::DownSampleRate],linewidth= linewidth)
    AxonConcentrationLines=plt.plot(TracetInS, TraceWeighterInAxonConcentration,linewidth= linewidth)
    plt.legend([ConcentrationLines,AxonConcentrationLines], [labels,'Axon'])
    plt.xlabel('Time (s)')
    plt.title('Receptor Concentration')
    
    X=TraceWeighters[StartStep:NumberOfSteps,0][::DownSampleRate]
    Y=TraceWeighters[StartStep:NumberOfSteps,1][::DownSampleRate]
    Z=TraceWeighters[StartStep:NumberOfSteps,2][::DownSampleRate]
    
    figure4 = plt.figure()
    plt.plot(X,Y)
    plt.xlabel('Time (s)')
    plt.title('2 Instantaneous Synaptic Strength')
    plt.xlabel('Instantaneous Synaptic Strength 0')
    plt.ylabel('Instantaneous Synaptic Strength 1')    
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
    
    figure6 = plt.figure()
    figure6lines = plt.plot(TracetInS[::DownSampleRate], TraceWeightersCentral[::DownSampleRate], label=labels, linewidth= linewidth)
    plt.legend(figure6lines, labels)
    plt.title('Center of Synaptic Strength Oscillation')
    plt.xlabel('Time (s)')
    
    figure7 = plt.figure()
    figure7lines = plt.plot(TracetInS[::DownSampleRate], TraceWeighterVarDamping[::DownSampleRate], label=labels, linewidth= linewidth)
    plt.legend(figure7lines, labels)
    plt.title('Damping factor')   
    plt.xlabel('Time (s)')
    
    figure8 = plt.figure()
    figure8lines = plt.plot(TracetInS[::DownSampleRate], TraceEquivalentVolume[::DownSampleRate], label=labels, linewidth= linewidth)
    plt.legend(figure7lines, labels)
    plt.xlabel('Time (s)')
    plt.title('Receptor Storage Capacity') 
#%
    figure9 = plt.figure()
    figure9ax1 = figure9 .add_subplot(111)  
    points0,points1 = CrossAnalysis(TraceWeighters[:,0],TraceWeightersCentral[:,0],TraceWeighters,TracetInS)
    if FullScale:
        figure9ax1.set_xlim(0,1)
        figure9ax1.set_ylim(0,1)
#    points0={'t':[],'points':[]}
#    points1={'t':[],'points':[]}
#    GreaterThanCentre=(TraceWeighters[0,0]>TraceWeightersCentral[0,0])
#    print(TraceWeighters[0,0])
#    print(TraceWeightersCentral[0,0])
#    for i1 in range(len(TraceWeighters)):
##        print(TraceWeighters[i1,0])
##        print(TraceWeightersCentral[i1,0])
#        if GreaterThanCentre == True:
#            if TraceWeighters[i1,0]<TraceWeightersCentral[i1,0]:
#                #print(TraceWeighters[i1,0])
#                points0['points'].append(TraceWeighters[i1])
#                points0['t'].append(TracetInS[i1])
#                GreaterThanCentre = False
#        elif GreaterThanCentre ==  False:
#            if TraceWeighters[i1,0]>TraceWeightersCentral[i1,0]:
#                #print(TraceWeighters[i1,0])
#                points1['points'].append(TraceWeighters[i1])
#                points1['t'].append(TracetInS[i1])
#                GreaterThanCentre = True
#    #c = np.empty(len(m[:,0])); c.fill(megno)
#    points0['points']=np.array(points0['points'])
#    points1['points']=np.array(points1['points'])
    print('points0')
    print(points0['points'])
    print('points1')
    print(points1['points'])
    
    pointsploted0 = figure9ax1.scatter(points0['points'][:,1],points0['points'][:,2],c=points0['t'], cmap=plt.cm.get_cmap('Greens'), marker=".", edgecolor='none') #c=c, ,  cmap=cm
    pointsploted1 = figure9ax1.scatter(points1['points'][:,1],points1['points'][:,2],c=points1['t'], cmap=plt.cm.get_cmap('Blues'), marker=".", edgecolor='none')
    #plt.legend(figure7lines, labels)
    plt.colorbar(pointsploted0)
    plt.colorbar(pointsploted1)
    plt.title('Poincare map') 
    plt.xlabel('Instantaneous Synaptic Strength 1')
    plt.ylabel('Instantaneous Synaptic Strength 2')
 #%   
    figure1.tight_layout()
    figure2.tight_layout()
    figure3.tight_layout()
    figure4.tight_layout()
    figure5.tight_layout()
    figure6.tight_layout()
    figure7.tight_layout()
    figure8.tight_layout()
    figure9.tight_layout()
    if savePlots == True:
        pp = PdfPages(path+"DynamicSynapse"+TimOfRecording+'.pdf')
        figure1.savefig(pp, format='pdf')
        figure2.savefig(pp, format='pdf')
        figure3.savefig(pp, format='pdf')
        figure4.savefig(pp, format='pdf')
        figure5.savefig(pp, format='pdf')
        figure6.savefig(pp, format='pdf')
        figure7.savefig(pp, format='pdf')
        figure8.savefig(pp, format='pdf')
        figure9.savefig(pp, format='pdf')
        pp.close()
#        Figures = {'TraceWeighters':figure1, 'TraceWeighterVarRates':figure2, 'TraceWeighterInAxon':figure3, '2TraceWeighters':figure4, '3DTraceWeighters':figure5, 'WeightersCentral':figure6, 'Damping':figure7,'EquivalentVolume':figure8,'Poincare map':figure9}
#        with open(path+"DynamicSynapse"+TimOfRecording+'.pkl', 'wb') as pkl:
#            dill.dump(Figures, pkl)

            
    return [figure1,figure2,figure3,figure4,figure5, figure6, figure7, figure8,figure9, ax]
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
#    Tracet, TraceWeighters, TraceWeighterVarRates, TraceWeighterInAxon, traceWeightersCentral,traceWeighterVarDamping 
    NeuonNumber=1
    newSlice= [slice(None)]*3
    newSlice[1]=NeuonNumber
    Traces = Tracet, ADSA.Trace['Weighters'][newSlice], ADSA.Trace['WeighterVarRates'][newSlice], ADSA.Trace['WeighterInAxon'][newSlice], ADSA.Trace['WeightersCentral'][newSlice], ADSA.Trace['WeighterVarDamping'][newSlice], ADSA.Trace['EquivalentVolume'][newSlice]
  

    return ADSA, Traces            
            
if __name__=="__main__":
    InintialDS =1
    SearchParameters =0
    InintialSearch =1
    SingleSimulation = 1
    PlotPhasePotrait = 1
    TimOfRecording=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    if InintialDS:
        NumberOfNeuron=1
        NumberOfSynapses = 1# N =3 tauWV =50; #N = 6 tauWV = 25
        Weighters= 0.25#np.random.rand(NumberOfNeuron,NumberOfSynapses) #0.5*np.ones(NumberOfSynapses) +0.001*np.random.rand(NumberOfSynapses)  #
        WeighteAmountPerSynapse = 1
        WeighterInAxon = WeighteAmountPerSynapse* NumberOfSynapses - Weighters
        WeighterVarRates = 0#np.zeros((NumberOfNeuron,NumberOfSynapses))
    
        
        
    #    TraceWeighters = np.zeros((NumberOfSteps,NumberOfNeuron,NumberOfSynapses))
    #    TraceWeighterVarRates  = np.zeros((NumberOfSteps,NumberOfNeuron,NumberOfSynapses))
    #    TraceWeighterInAxon = np.zeros((NumberOfSteps,NumberOfNeuron))
#        CW = 100
#        tauWV =60#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#130   100
#        rWV = 7000 #7000
#        scale=1
#        damping =2
#        CW = 100
#        SimulationTimeInterval = 30
#        CW = 100

### ratio of intergration of postive value oscillation and nagative value oscillation is low
#        tauWV =500#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 170#130   100
#        rWV = 7000 #7000
#        scale=1
#        damping =2
#        CW = 100
#        SimulationTimeInterval = 10
## oscillation with periods of 300 to 500 seconds 
#
#        tauWV =0.1#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 34#170#130   100
#        rWV = 7000*500*100#7000
#        scale=1
#        damping =2*7000
#        CW = 100
#        SimulationTimeInterval = 100
#        
## oscillation with periods of 20seconds  *** when receptor amount 10

#        tauWV =0.1#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
#        aWV = 100#170#130   100
#        rWV = 7000*500*300#7000
#        scale=1
#        damping =2*3000
#        CW = 100
## oscillation with periods of 50seconds  *** when receptor amount 1
#
        tauWV =0.5#40#19#17.8#40#35#50#40#17.8#40 if flow rate times w*v choose 40, if flow rate  times w or v choose 20
        aWV = 100#170#130   100
        rWV = 7000*500#7000
        scale=1
        damping =2*7000
        CW = 100
        SimulationTimeInterval = 100
        
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
        WeightersCentral = 0.5#(NumberOfSynapses) #* np.random.rand(NumberOfNeuron,NumberOfSynapses) # 0.6 * np.random.rand(NumberOfNeuron,NumberOfSynapses) #np.array([4, 1, 1, 1, 1])#np.ones(NumberOfSynapses)*0.4 + 0.3 * np.random.rand(NumberOfSynapses) #np.ones(NumberOfSynapses)/2 + [0.8, 0.1,0.1, 0.1, 0]  #[0.2, 0.1, 0]
        WeighterVarDamping =  damping #np.array([10, 2, 2, 2, 2]) #[10,2,2,2,2]       #[2,2,2]
#        WeighterVarDamping[0,1] = 4
        Parameters = [CW, tauWV, aWV , rWV, WeightersCentral,WeighterVarDamping]
        
        ADSA=DynamicSynapseArray( NumberOfSynapses = [NumberOfNeuron, NumberOfSynapses], CW = CW, tauWV = tauWV, aWV = aWV, rWV = rWV,scale=scale, \
                    WeightersCentral = WeightersCentral , WeighterVarDamping = WeighterVarDamping, WeighteAmountPerSynapse = WeighteAmountPerSynapse, \
                    Weighters = Weighters, WeighterVarRates = WeighterVarRates,WeightersCentralUpdateCompensate = 0)
#        ADSA.DampingUpdateRate=0
        SimulationTimeLenth = 60*60*1000
        
        dt = SimulationTimeInterval 
        NumberOfSteps = int(SimulationTimeLenth/SimulationTimeInterval)
        Tracet = np.zeros(NumberOfSteps)
        ADSA.InitRecording(NumberOfSteps)  
    if SearchParameters :
        if InintialSearch:
            searchSamples=[10,10]
            centralSearchSamples=np.floor(np.array(searchSamples)/2).astype(int)
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
            randSearchRate0[centralSearchSamples[0]]=0
            randSearchRate1=(np.random.random_sample(Arg1space.shape)-0.5)*0#.1
            randSearchRate0[centralSearchSamples[1]]=0
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
#            ADSA.StepSynapseDynamics( SimulationTimeInterval,0)
            ADSA.StepSynapseDynamics( SimulationTimeInterval,0)

            if  ADSA.RecordingState:
                ADSA.Recording()  
                Tracet[step] = step*SimulationTimeInterval
                #%
            if step % 1000 == 0:
                print('%d of %d steps'%(step,NumberOfSteps))
        
    #    Tracet, TraceWeighters, TraceWeighterVarRates, TraceWeighterInAxon, traceWeightersCentral,traceWeighterVarDamping 
        NeuonNumber=0
        newSlice= [slice(None)]*3
        newSlice[1]=NeuonNumber
        Traces = Tracet, ADSA.Trace['Weighters'][newSlice], ADSA.Trace['WeighterVarRates'][newSlice], ADSA.Trace['WeighterInAxonConcentration'][newSlice], ADSA.Trace['WeightersCentral'][newSlice], ADSA.Trace['WeighterVarDamping'][newSlice], ADSA.Trace['EquivalentVolume'][newSlice]
#%%
        UpHalfWeightSum= (ADSA.Trace['Weighters'][newSlice]-WeightersCentral)[ADSA.Trace['Weighters'][newSlice]>WeightersCentral].sum()
        UpHalfTime=ADSA.Trace['Weighters'][newSlice][ADSA.Trace['Weighters'][newSlice]>WeightersCentral].shape[0]*dt
        DownHalfWeightSum= (WeightersCentral-ADSA.Trace['Weighters'][newSlice])[ADSA.Trace['Weighters'][newSlice]<WeightersCentral].sum()
        DownHalfTime=ADSA.Trace['Weighters'][newSlice][ADSA.Trace['Weighters'][newSlice]<WeightersCentral].shape[0]*dt
        print("UpHalfWeightSum:  %f, UpHalfTime:  %f"%(UpHalfWeightSum, UpHalfTime))
        print("DownHalfWeightSum:%f, DownHalfTime:%f"%(DownHalfWeightSum, DownHalfTime))
        print("UDWeightRate:%f, UDTimeRate:%f"%(UpHalfWeightSum/DownHalfWeightSum, float(UpHalfTime)/DownHalfTime))
#%%
#        figure1,figure2,figure3,figure4,figure5, figure6, figure7,figure8,figure9,ax = plot(TimOfRecording, Traces, path='/media/archive2T/chitianqilin/SimulationResult/DynamicSynapse/Plots/', savePlots=True, linewidth= 0.2) #path=
        
        #%%
    #    f = open("I:/OneDrive - University of Edinburgh/Documents/MushroomBody Model/DynamicSynapse/DynamicSynapse"+TimOfRecording+'.pkl', 'wb')
    #    
    #    dill.dump(figure5,f  )
        #%%
    #    for angle in range(0, 360):
    #        ax.view_init(30, angle)
    #        #plt.draw()
    #        plt.pause(.0001)
    if PlotPhasePotrait :
        ViewScale=40
        PhasePotraitfig, PhasePotraitax = ADSA.PlotPhasePortrait( xlim=[0,1], ylim=[-0.00010, 0.00010], fig=None, ax=None, inputs=[WeighteAmountPerSynapse- WeightersCentral, WeightersCentral], Parameters=None)

 #       PhasePotraitfig, PhasePotraitax = ADSA.PlotPhasePortrait( xlim=[0.0,1.5], ylim=[-0.000005*ViewScale, 0.000005*ViewScale], fig=None, ax=None, inputs=[1, WeightersCentral], Parameters=None)
        #PhasePotraitax.plot(ADSA.Trace['Weighters'][:,0,0],ADSA.Trace['WeighterVarRates'][:,0,0])
        points = np.array([ADSA.Trace['Weighters'][:,0,0], ADSA.Trace['WeighterVarRates'][:,0,0]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        DistancePerStep=np.linalg.norm(points[1:]- points[:-1], axis=2).ravel()
        norm = mpl.colors.Normalize(vmin=DistancePerStep.min(), vmax=DistancePerStep.max(), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hot)
        color=mapper.to_rgba(DistancePerStep)
        lc = LineCollection(segments, cmap=plt.get_cmap('copper'),colors=color)
        PhasePotraitax.add_collection(lc)
        
