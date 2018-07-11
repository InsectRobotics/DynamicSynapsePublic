#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:01:23 2017

@author: chitianqilin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import dill
class LIFNeuronArray:
    def __init__(self, NumberOfNuerons=4, VReset=-60, VFire=0, Vinit=-60, Capacity=1, LeakConduction=0.001, ISynapse=0):
        self.NumberOfNuerons= NumberOfNuerons
        if type(VReset) is list or type(VReset) is np.ndarray:
            self.VReset = np.array(VReset)
        else:
            self.VReset = np.ones(NumberOfNuerons)*VReset
            
        if type(VFire) is list or type(VFire) is np.ndarray:
            self.VFire = np.array(VFire)
        else:
            self.VFire = np.ones(NumberOfNuerons)*VFire
           
        if type(Capacity) is list or type(Capacity) is np.ndarray:
            self.Capacity = np.array(Capacity)
        else:
            self.Capacity = np.ones(NumberOfNuerons)*Capacity
            
        if type(LeakConduction) is list or type(LeakConduction) is np.ndarray:
            self.LeakConduction = np.array(LeakConduction)
        else:
            self.LeakConduction = np.ones(NumberOfNuerons)*LeakConduction

        if type(ISynapse) is list or type(ISynapse) is np.ndarray:
            self.ISynapse = np.array(ISynapse)
        else:
            self.ISynapse = np.ones(NumberOfNuerons)*ISynapse
            
        if type(Vinit) is list or type(Vinit) is np.ndarray:
            self.V = np.array(Vinit)
        else:
            self.V = np.ones(NumberOfNuerons)*Vinit            
            
        self.ILeak = np.zeros(NumberOfNuerons) 
        self.Spiking = np.zeros(NumberOfNuerons, dtype=bool)
        self.Reset = np.zeros(NumberOfNuerons, dtype=bool)
        self.IPost = np.zeros(NumberOfNuerons, dtype=bool)
        
    def step(self, dt, IPost):
        self.dt = dt
        if type(IPost) is list or type(IPost) is np.ndarray:
            self.IPost = np.array(IPost)
        else:
            self.IPost = np.ones(self.NumberOfNuerons)*IPost   
        self.Reset = np.zeros(self.NumberOfNuerons, dtype=bool)
        self.Reset[self.Spiking] = True
        self.V[self.Spiking] = self.VReset[self.Spiking]
        self.Spiking = np.zeros(self.NumberOfNuerons, dtype=bool)
        self.ILeak = ( (self.VReset-self.V) * self.LeakConduction)
        #print(IPost, ILeak, Capacity)
        self.V += (IPost + self.ILeak )/self.Capacity*dt #mv
        self.Spiking = np.greater_equal( self.V ,self.VFire)


    def Record(self):     
        if self.RecordingState:
            self.RecordingInPointer+=1
            if self.recordCycling:
                self.RecordingOutPointer += 1
                if self.RecordingOutPointer == self.RecordingLenth:
                    self.RecordingOutPointer = 0
            if self.RecordingInPointer == self.RecordingLenth:
                self.RecordingInPointer = 0
                self.RecordingOutPointer = 1
                self.recordCycling = 1
            self.Trace['t'][self.RecordingInPointer] = self.Trace['t'][self.RecordingInPointer-1]+self.dt
            self.Trace['V'][self.RecordingInPointer,:] = self.V 
            self.Trace['ILeak'][self.RecordingInPointer,:] = self.ILeak
            self.Trace['Spiking'][self.RecordingInPointer,:] = self.Spiking
            self.Trace['Reset'][self.RecordingInPointer,:] = self.Reset  
            self.Trace['IPost'][self.RecordingInPointer,:] = self.IPost
    def InitRecording(self, lenth):
        self.RecordingState = True
        self.RecordingLenth = lenth
        self.RecordingInPointer = 0
        self.RecordingOutPointer = 0
        self.recordCycling = 0
        self.Trace = {'t': np.zeros(lenth), \
                        'V':np.zeros(np.append(lenth,self.NumberOfNuerons).ravel()), \
                        'ILeak' : np.zeros(np.append(lenth,self.NumberOfNuerons).ravel()), \
                        'IPost' : np.zeros(np.append(lenth,self.NumberOfNuerons).ravel()), \
                        'Spiking' : np.zeros(np.append(lenth,self.NumberOfNuerons).ravel(),dtype = bool),\
                        'Reset' : np.zeros(np.append(lenth,self.NumberOfNuerons).ravel(),dtype = bool) \
                        }
    def Plot(self,TimOfRecording='', path='', savePlots=False, linewidth= 1):
        labels = [str(i) for i in range(self.NumberOfNuerons)]
        self.figure1=plt.figure()
        VLine=plt.plot(self.Trace['t']/1000,self.Trace['V'], linewidth= linewidth)
        plt.legend(VLine, labels)
        plt.title('Membrane potential')
        plt.ylabel('Membrane potential (mV)')
        plt.xlabel('Time (s)')
        self.figure2=plt.figure()
        ILeakLine=plt.plot(self.Trace['t']/1000,self.Trace['ILeak'], linewidth= linewidth) 
        plt.ylabel('Membrane potential (pA)')
        plt.xlabel('Time (s)')
        plt.legend(ILeakLine, labels)
        plt.title('Leak current')
        if savePlots == True:
            pp = PdfPages(path+"LIFNPlots"+TimOfRecording+'.pdf')
            self.figure1.savefig(pp, format='pdf')
            self.figure1.savefig(pp, format='pdf')
            pp.close()
            Figures = {'V':self.figure1, 'ILeak':self.figure2}
            with open(path+"LIFNPlots"+TimOfRecording+'.pkl', 'wb') as pkl:
                dill.dump(Figures, pkl)
        return self.figure1, self.figure2
if __name__=='__main__':
    ALIFNeuronArray=LIFNeuronArray()
    TimeLength=100000
    dt=33
    steps=int(TimeLength/dt)
    ALIFNeuronArray.InitRecording(steps)
    for step in range(steps):
        ALIFNeuronArray.step(dt, 0.1)
    ALIFNeuronArray.Plot()    
