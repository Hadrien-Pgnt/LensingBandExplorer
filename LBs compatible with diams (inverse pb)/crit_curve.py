#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:26:08 2021

Author: Hadrien Paugnat

This code computes diameter values for Kerr critical curves 
(for a single spin & incl value, or on a grid),
and perform phoval fits for these critical curves (with or without a diameter constraint)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def phoval(phi, R0, R1, R2, chi, X):
    return ( R0 + np.sqrt(R1**2 * np.sin(phi)**2 + R2**2 * np.cos(phi)**2) + (X -chi)*np.cos(phi) + np.arcsin(chi*np.cos(phi)))

def phoval_points(phi, phi0, R0, R1, R2, chi, X):
    ''' Returns the (alpha, beta) points for the (rotated) phoval of params (phi0, R0, R1, R2, chi, X)
    /!\ phi does not correspond to the polar coordinate (sigma), but rather to some intrinsic parametrization of the phoval'''
    f = ( R0 + np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) + (X -chi)*np.cos(phi-phi0) + np.arcsin(chi*np.cos(phi-phi0)))
    fprime = (R1**2-R2**2)*np.sin(phi-phi0)*np.cos(phi-phi0)/np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) - (X -chi)*np.sin(phi-phi0) - chi*np.sin(phi-phi0)/np.sqrt(1-(chi*np.cos(phi-phi0))**2)
    x = f*np.cos(phi) - fprime *np.sin(phi)
    y = f*np.sin(phi) + fprime *np.cos(phi)
    return x+y*1j


n_points = int(1e5)
def crit_curve(a, incl):
    ''' Returns the orthogonal (minimal) and parallel (maximal) diameters - in units of M -
    of the critical curve for a Kerr black hole of spin param a and inclination incl'''
    
    thetaobs = incl*math.pi/180 #in radians
    
    # print('Generating data for spin a=',a,'; inclination i=', incl)
    rplus = 2*(1+ math.cos(2/3 * math.acos(a))) 
    rminus = 2*(1+ math.cos(2/3 * math.acos(-a)))
        
    r = np.linspace(rplus, rminus, n_points)
    
    delta = r**2 - 2*r + a**2
    diff = 2*np.sqrt(delta*(2*r**3-3*r**2+a**2))
    uplus = r/(a**2 * (r-1)**2) * (-r**3+3*r-2*a**2 + diff)
    uminus = r/(a**2 * (r-1)**2) * (-r**3+3*r-2*a**2 - diff)
    
    l = (r**2-a**2 - r*delta)/ a / (r-1)
    
    rhoD = np.sqrt(a**2 * ( math.cos(thetaobs)**2 - uplus * uminus) + l**2)
    cos_phi_p = -l/ (rhoD*math.sin(thetaobs))

    x = rhoD[abs(cos_phi_p)<1]*cos_phi_p[abs(cos_phi_p)<1]
    y = rhoD[abs(cos_phi_p)<1]*np.sqrt(1-cos_phi_p[abs(cos_phi_p)<1]**2)
    
    d_ortho= np.max(x) - np.min(x) 
    d_parallel= 2*np.max(y)  #we only plotted the half-circle here, but the figure is symmetric along the x axis
    
    #d_ortho (resp. d_parallel) is the minimal (resp. maximal) diameter for the critical curve  
    return (d_parallel, d_ortho)

class CritCurveGrid:
    
    def __init__(self, n_grid):
        self.n_grid = n_grid #size of the grid
        
        self.spins = np.linspace(0.01,0.999,self.n_grid)
        self.inclinations = np.linspace(0.01,90,self.n_grid)
        self.a, self.incl = np.meshgrid(self.spins,self.inclinations)

    def generate_map(self):
        '''Computes the values of (d+, d-) of the critical curves 
        for a grid of points (a,i) of size n_grid x n_grid  '''

        dparr, dortho = np.vectorize(crit_curve) (self.a,self.incl)
        
        np.save('crit_curve_map_(a,i).npy', np.array([self.spins, self.inclinations]), allow_pickle=True)
        np.save('crit_curve_map_(d+,d-).npy', np.array([dparr, dortho]), allow_pickle=True)


    def plot_map(self):
        ''' Plots the result of generate_grid if already computed (if not, computes it):
            the isocurves for d_parallel and the fractional asymmetry fA = 1-d_orthognoal/d_parallel
            in the (a,i) plane'''
        
        if os.path.exists('crit_curve_map_(d+,d-).npy'):
            
            try:
                diams = np.load('crit_curve_map_(d+,d-).npy')
                
                fA = 1-diams[1]/diams[0] #Fractional asymmetry
                
                fig, ax = plt.subplots()
                ax.set_xlabel("Spin parameter")
                ax.set_ylabel("Inclination (°)")
                ax.annotate('Fractional asymmetry',(0.22,70), color = 'r', fontsize = 20)
                ax.contour(self.a, self.incl, fA,  np.arange(0,0.1,0.002), colors='r', linewidths = 0.1)
                CS = ax.contour(self.a, self.incl, fA,  np.arange(0,0.1,0.01), colors='r')
                ax.clabel(CS, manual=False, fmt ='%1.2f', inline_spacing = 1)
                
                ax.annotate(r'$d_{\parallel}$(M)',(0.4,30), color = 'b', fontsize = 20)
                ax.contour(self.a, self.incl, diams[0],  np.arange(9.7,10.4,0.02), colors='b', linewidths = 0.1)
                C = ax.contour(self.a, self.incl, diams[0],  np.arange(9.7,10.4,0.1), colors='b')
                
                ax.clabel(C, manual=False, fmt ='%1.1f', inline_spacing = 1)
                
            except TypeError:
                print('Size of precomputed grid does not match the wanted size. New computation in progress...')
                self.generate_map()
                self.plot_map()
        
        else:
            self.generate_map()
            self.plot_map()


def crit_curve_phoval_fit(a,incl, plot = False):
    ''' Returns the best-fit params for a phoval fit of
    the critical curve for a Kerr black hole of spin param a and inclination incl'''
    
    thetaobs = incl*math.pi/180
    
    rplus = 2*(1+ math.cos(2/3 * math.acos(a))) 
    rminus = 2*(1+ math.cos(2/3 * math.acos(-a)))
        
    r = np.linspace(rplus, rminus, n_points)
    
    delta = r**2 - 2*r + a**2
    diff = 2*np.sqrt(delta*(2*r**3-3*r**2+a**2))
    uplus = r/(a**2 * (r-1)**2) * (-r**3+3*r-2*a**2 + diff)
    uminus = r/(a**2 * (r-1)**2) * (-r**3+3*r-2*a**2 - diff)

    l = (r**2-a**2 - r*delta)/ a / (r-1)
    
    rhoD = np.sqrt(a**2 * ( math.cos(thetaobs)**2 - uplus * uminus) + l**2)
    cos_phi_p = -l/ (rhoD*math.sin(thetaobs))
    
    phi_p = np.arccos(cos_phi_p[abs(cos_phi_p)<1])
    
    phi_p = np.concatenate((np.flip(-phi_p), phi_p))
    rhoD = np.concatenate((np.flip(rhoD[abs(cos_phi_p)<1]), rhoD[abs(cos_phi_p)<1]))
    xcurve = rhoD * np.cos(phi_p)
    ycurve = rhoD * np.sin(phi_p)
    
    phi_param = np.arctan(-np.diff(xcurve)/np.diff(ycurve))
    f = xcurve[:-1]* np.cos(phi_param) + ycurve[:-1]* np.sin(phi_param)
    
    #reordering and = mod pi to have phi in [0,2pi[ and in growing order
    phi_param,f = np.concatenate((phi_param[(f>0) & (phi_param>0)],phi_param[(f <0)& (phi_param<0)]+np.pi,phi_param[(f <0)& (phi_param>0)]+np.pi,phi_param[(f > 0) & (phi_param<0)]+2*np.pi)), np.concatenate((f[(f > 0) & (phi_param>0)],-f[(f <0)& (phi_param<0)],-f[(f <0)& (phi_param>0)],f[(f > 0) & (phi_param<0)]))
    
    popt, pcov = curve_fit(phoval, phi_param, f, p0 = np.array([4.5,0.6,0.06,0.98,2]), bounds=([0,0,0,-1,-np.inf],[np.inf,np.inf,np.inf,1,np.inf]))
    
    if plot :
        plt.figure()
        ax = plt.subplot(111, polar=False)
        ax.plot(xcurve, ycurve,'r', label = 'critical curve')
    
        zopt = phoval_points(phi_p, 0.,*popt)
        ax.plot(np.real(zopt),np.imag(zopt), 'b:', label=r'numerical fit with : $R_0$=%5.3f, $R_1$=%5.3f, $R_2$=%5.3f, $\chi$=%5.3f, X=%5.3f' % tuple(popt))
       
    return popt


### TBD: implement what is described in the docstring
def crit_curve_phoval_fit_with_diam_constraint(a, incl, dplus, dminus, plot = False):
    ''' Returns the best-fit params for a phoval fit of
    the critical curve for a Kerr black hole of spin param a and inclination incl
    with the constraint that the phoval must have max/min diameters dplus/dminus'''
    
    thetaobs = incl*math.pi/180
    
    rplus = 2*(1+ math.cos(2/3 * math.acos(a))) 
    rminus = 2*(1+ math.cos(2/3 * math.acos(-a)))
        
    r = np.linspace(rplus, rminus, n_points)
    
    delta = r**2 - 2*r + a**2
    diff = 2*np.sqrt(delta*(2*r**3-3*r**2+a**2))
    uplus = r/(a**2 * (r-1)**2) * (-r**3+3*r-2*a**2 + diff)
    uminus = r/(a**2 * (r-1)**2) * (-r**3+3*r-2*a**2 - diff)

    l = (r**2-a**2 - r*delta)/ a / (r-1)
    
    rhoD = np.sqrt(a**2 * ( math.cos(thetaobs)**2 - uplus * uminus) + l**2)
    cos_phi_p = -l/ (rhoD*math.sin(thetaobs))
    
    phi_p = np.arccos(cos_phi_p[abs(cos_phi_p)<1])
    
    phi_p = np.concatenate((np.flip(-phi_p), phi_p))
    rhoD = np.concatenate((np.flip(rhoD[abs(cos_phi_p)<1]), rhoD[abs(cos_phi_p)<1]))
    xcurve = rhoD * np.cos(phi_p)
    ycurve = rhoD * np.sin(phi_p)
    
    phi_param = np.arctan(-np.diff(xcurve)/np.diff(ycurve))
    f = xcurve[:-1]* np.cos(phi_param) + ycurve[:-1]* np.sin(phi_param)
    
    #reordering and = mod pi to have phi in [0,2pi[ and in growing order
    phi_param,f = np.concatenate((phi_param[(f>0) & (phi_param>0)],phi_param[(f <0)& (phi_param<0)]+np.pi,phi_param[(f <0)& (phi_param>0)]+np.pi,phi_param[(f > 0) & (phi_param<0)]+2*np.pi)), np.concatenate((f[(f > 0) & (phi_param>0)],-f[(f <0)& (phi_param<0)],-f[(f <0)& (phi_param>0)],f[(f > 0) & (phi_param<0)]))
    
    def constrained_phoval(R0,phi0 ):
        pass
    
    popt, pcov = curve_fit(phoval, phi_param, f, p0 = np.array([4.5,0.6,0.06,0.98,2]), bounds=([0,0,0,-1,-np.inf],[np.inf,np.inf,np.inf,1,np.inf]))
    
    if plot :
        plt.figure()
        ax = plt.subplot(111, polar=False)
        ax.plot(xcurve, ycurve,'r', label = 'critical curve')
    
        zopt = phoval_points(phi_p, 0.,*popt)
        ax.plot(np.real(zopt),np.imag(zopt), 'b:', label=r'numerical fit with : $R_0$=%5.3f, $R_1$=%5.3f, $R_2$=%5.3f, $\chi$=%5.3f, X=%5.3f' % tuple(popt))
       
    return popt
    
