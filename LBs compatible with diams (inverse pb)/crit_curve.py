#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:26:08 2021

@author: stage
"""
import numpy as np
import math
import matplotlib.pyplot as plt


n_points = int(1e5)
def crit_curve(a,thetaobs):  #thetaobs in radians
    ''' Returns the orthogonal (minimal) and parallel (maximal) diameters - in units of M -
    of the critical curve for a Kerr black hole of spin param a and inclination thetaobs'''
    
    # print('Generating data for spin a=',a,'; inclination i=', thetaobs)
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

# print(crit_curve(0.94, 17*math.pi/180))

n_grid = 200
    
spins = np.linspace(0.01,0.999,n_grid)
inclinations = np.linspace(0.01,math.pi/2,n_grid)

a,thetaobs = np.meshgrid(spins,inclinations)

# dparr, dortho = np.vectorize(crit_curve) (a,thetaobs)

# np.save('crit_curve_map_(a,i).npy', np.array([spins, inclinations]), allow_pickle=True)
# np.save('crit_curve_map_(d+,d-).npy', np.array([dparr, dortho]), allow_pickle=True)

### Plot isocontours 
diams = np.load('crit_curve_map_(d+,d-).npy')
fA = 1-diams[1]/diams[0] #Fractional asymmetry

fig, ax = plt.subplots()
ax.set_xlabel("Spin parameter")
ax.set_ylabel("Inclination (°)")
ax.annotate('Fractional asymmetry',(0.22,70), color = 'r', fontsize = 20)
CSsub = ax.contour(a, thetaobs*180/np.pi, fA,  np.arange(0,0.1,0.002), colors='r', linewidths = 0.1)
CS = ax.contour(a, thetaobs*180/np.pi, fA,  np.arange(0,0.1,0.01), colors='r')
ax.clabel(CS, manual=False, fmt ='%1.2f', inline_spacing = 1)

ax.annotate(r'$d_{\parallel}$(M)',(0.4,30), color = 'b', fontsize = 20)
Csub = ax.contour(a, thetaobs*180/np.pi, diams[0],  np.arange(9.7,10.4,0.02), colors='b', linewidths = 0.1)
C = ax.contour(a, thetaobs*180/np.pi, diams[0],  np.arange(9.7,10.4,0.1), colors='b')

ax.clabel(C, manual=False, fmt ='%1.1f', inline_spacing = 1)


