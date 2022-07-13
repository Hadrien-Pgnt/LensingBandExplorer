#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:41:48 2021

Author: Hadrien Paugnat

This code computes the lensings bands (LBs) compatible with a fixed point in the (d+,d-) plane, assumed to be a phoval
"""

from lensingbands import LensingBand

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, root
from scipy import optimize, interpolate
# from scipy.spatial import ConvexHull
import random as rnd
import os

## For nicer plots

from matplotlib import rcParams
rcParams['font.family']='serif'
#rcParams['text.usetex']=True 

red = [228/255, 26/255, 28/255]
gold = [255/255, 215/255, 0/255] 
blue = [55/255, 126/255, 184/255]
green = [77/255, 175/255, 74/255]
purple = [152/255, 78/255, 163/255]
orange = [255/255, 127/255, 0/255]


########################
''''Useful functions and constants'''

def point_in_convex_hull(point, hull, tolerance=1e-12):
    '''Useful function to check if a point is in a ConvexHull'''
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def phoval_points(phi, phi0, R0, R1, R2, chi, X):
    ''' Returns the (alpha, beta) points for the (rotated) phoval of params (phi0, R0, R1, R2, chi, X)
    /!\ phi does not correspond to the polar coordinate (sigma), but rather to some intrinsic parametrization of the phoval'''
    f = ( R0 + np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) + (X -chi)*np.cos(phi-phi0) + np.arcsin(chi*np.cos(phi-phi0)))
    fprime = (R1**2-R2**2)*np.sin(phi-phi0)*np.cos(phi-phi0)/np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) - (X -chi)*np.sin(phi-phi0) - chi*np.sin(phi-phi0)/np.sqrt(1-(chi*np.cos(phi-phi0))**2)
    x = f*np.cos(phi) - fprime *np.sin(phi)
    y = f*np.sin(phi) + fprime *np.cos(phi)
    return x+y*1j

########################

class DiameterMeasurement:
    ''' To compute the values of spin & inclination for which the corresponding lensing bands (for a given order)
    contains a phoval with extremal diameters d+ and d- (fixed values) 
    This case corresponds to a measurement of a photon ring (and its angle-dependant diameter) of a Kerr black hole with unknown parameters'''
    def __init__(self, dplus, dminus, order, NN, spinguess, inclguess):
        self.dplus = dplus #maximal diameter measured for the photon ring
        self.dminus = dminus #minimal diameter measured for the photon ring
        self.order = order # order of the ring (we are typically interested in the n=2 ring)
        self.NN = NN #nb of points computed for the LB edges 
        
        self.phoval_params = np.zeros(6) #rotated phoval parameters   
        
        self.spin_guess = spinguess
        self.incl_guess = inclguess
        
        self.address = os.path.abspath(__file__).replace('\\','/')
        
        ##Creates a directory 'Edge_data' if not already existing
        self.edge_data_dir = '/'.join(self.address.split("/")[:-1]) + '/Edge_data'
        if not os.path.exists(self.edge_data_dir):
            os.makedirs(self.edge_data_dir)
            
        ##Creates a directory 'Step_data' if not already existing
        self.step_data_dir = '/'.join(self.address.split("/")[:-1]) + '/Steps_data'
        if not os.path.exists(self.step_data_dir):
            os.makedirs(self.step_data_dir)
            
        # Loads the grid of spins & inclinations along with the associated critical curve diameter data computed using crit_curve.py
        spins = np.load('crit_curve_map_(a,i).npy')[0]
        incls = np.load('crit_curve_map_(a,i).npy')[1]
        diams = np.load('crit_curve_map_(d+,d-).npy')
        
        # interpolate to obtain a (d+,d-) -> (a,i) map for the critical curves
        self.d_parr = interpolate.interp2d(spins, incls, diams[0], kind='cubic')
        self.d_ortho = interpolate.interp2d(spins, incls, diams[1], kind='cubic')
        
            
    def guess_spin_incl_from_crit_curve(self):
        ''' Computes the value of spin & inclination for which the max/min diameters of critical curve
        correspond to the "measured" (d+,d-)
        i.e. performs spin & inclination measurement by assimilating the photon ring to the critical curve'''
        
        def to_solve(x):
                return [*self.d_parr(x[0],x[1])-self.dplus, *self.d_ortho(x[0],x[1])-self.dminus]
        sol = root(to_solve, [self.spin_guess,self.incl_guess*math.pi/180], tol=1e-10)
        
        self.spin_guess = sol.x[0]
        self.incl_guess = sol.x[1]*180/math.pi
        
        

### TBD: adapt the code after this point 
### TBD: check that to_solve does not need a better x0 guess  
        
    def explore_at_step(self, step, incr, startpoint, nhops, tol, Ncheck):
        '''Loads the values for accepted/rejected pts in the (d+, d-) plane previoulsy aggregated during *step* steps
        Performs a new step (the *step+1*): a random walk starting from *startpoint*, 
        walking *nhops* times on a grid with a *incr* increment
        A point of the (d+, d-) plane is accepted if a phoval  - with these diameter values - is found that fits into the lensing band 
        (i.e. for which distance to the LB is =0, with a tolerance *tol* to account for numerical artifacts)'''
        
        data_load_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step)
        data_save_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step+1)
        
        if step!=0:
            #Loads the accepted/rejected values from the previous step
            accepted_base = np.load(data_load_path+'_accepted.npy')
            rejected_base = np.load(data_load_path+'_rejected.npy')
            
            if startpoint=='medium':
                # params of a "medium" phoval which we are sure lies in the lensing band
                params = 0.5*(np.array(self.phoval_outer.x)+np.array(self.phoval_inner.x))
                
            elif startpoint=='upper right':
                #The value in accepted_base which is the furthest upper right position in the (d+, d-) plane
                index=np.argmax(accepted_base[:,0]+accepted_base[:,1]) #index in accepted_base of the point in the upper right corner
                params=np.copy(accepted_base[index,2:]) # params of the rotated phoval corresponding to this point 
            
            elif startpoint=='lower right':
                #The value in accepted_base which is the furthest lower right position in the (d+, d-) plane
                index=np.argmax(accepted_base[:,0]-accepted_base[:,1]) #index in accepted_base of the point in the lower right corner
                params=np.copy(accepted_base[index,2:]) # params of the rotated phoval corresponding to this point 
            
            elif startpoint=='upper left':
                #The value in accepted_base which is the furthest upper left position in the (d+, d-) plane
                index=np.argmin(accepted_base[:,0]-accepted_base[:,1]) #index in accepted_base of the point in the upper left corner
                params=np.copy(accepted_base[index,2:]) # params of the rotated phoval corresponding to this point 
            
            elif startpoint=='lower left':
                #The value in accepted_base which is the furthest lower left position in the (d+, d-) plane
                index=np.argmin(accepted_base[:,0]+accepted_base[:,1]) #index in accepted_base of the point in the lower left corner
                params=np.copy(accepted_base[index,2:]) # params of the rotated phoval corresponding to this point 
            
            else:
                # Startpoint should be specified in the form of a string with coordinates d+ and d- separated by a ';'
                # (ex: '9.771;9.709')
                # Program will pick the accepted value closest to this point
                
                start_dplus = float(startpoint.split(";")[0])
                start_dminus = float(startpoint.split(";")[1])
                
                index=np.argmin(np.abs(start_dplus-accepted_base[:,0])+np.abs(start_dminus-accepted_base[:,1])) #index in accepted_base of the accepted point closest to the coords given in startpoint
                params=np.copy(accepted_base[index,2:]) # params of the rotated phoval corresponding to this point we selected
        
            rejected = list(rejected_base)
            accepted = list(accepted_base)
        
        else: #if step==0 (i.e. if nothing has already been computed)
            params = 0.5*(np.array(self.phoval_outer.x)+np.array(self.phoval_inner.x)) # medium by default
            accepted = []
            rejected = []
        
        ## Random walk

        for hop in range(nhops):
            print(hop)
            
            prev_params = np.copy(params)
            found_accepted = False
            already_tried =[]
            
            while not(found_accepted) and len(already_tried)<6:
                index_change = rnd.randint(1,3) #randomly pick R0,R1 or R2
                sign = 2*rnd.randint(0,1) - 1 # randomly pick +/-1
                
                if (index_change,sign) not in already_tried:
                    params[index_change] += sign*incr
                    
                    R0 = params[1]
                    R1 = max(params[2],params[3])
                    R2 = min(params[2],params[3])
                    d_plus = 2*(R0+R1)
                    d_minus = 2*(R0+R2)
                    # print(d_plus,d_minus,params)
                        
                    if self.dist_phoval_to_band(params,Ncheck)<= tol:
                        accepted.append([d_plus,d_minus,*params])
                        found_accepted = True
                    
                    else:
                        ## minimize over phi0, chi and X the distance to the lensing band
                        best_add_params = minimize(lambda p: self.dist_phoval_to_band([p[0],*params[1:4],p[1],p[2]],Ncheck), x0=([params[0],*params[4:]]),bounds=[(-math.pi,math.pi),(-1,1),(-np.inf,np.inf)])
                        bestparams = [best_add_params.x[0], *params[1:4], *best_add_params.x[1:]]

                        if self.dist_phoval_to_band(bestparams, Ncheck) <= tol:
                            params = bestparams
                            accepted.append([d_plus,d_minus,*params])
                            found_accepted = True
                            
                        else:                   
                            rejected.append([d_plus,d_minus,*params])
                            params=prev_params
                            already_tried.append((index_change,sign))
                            
        np.save(data_save_path+'_accepted.npy', accepted, allow_pickle=True)
        np.save(data_save_path+'_rejected.npy', rejected, allow_pickle=True)
    
    def explore_one_step(self, incr, startpoint, nhops, tol, Ncheck):
        ''' Determines the last step of exploration (0 if nothing was already computed)
        then launches explore_at_step at next step with the given parameters for the random walk  '''
        
        step_data_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)
        
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            os.makedirs(step_data_path)
            self.explore_at_step(0, incr, startpoint, nhops, tol, Ncheck)
        
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x_accepted.npy' or 'step_x_rejected.npy' where x is the nb of the step
            self.explore_at_step(laststep, incr, startpoint, nhops, tol, Ncheck)

    
    def plot_step(self, step):
        ''' Plots the results of the random walk after the given step'''
        
        if step==0:
            print('No accepted/rejected values computed yet')
        else:
            data_load_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step)
            accepted = np.load(data_load_path+'_accepted.npy')
            rejected = np.load(data_load_path+'_rejected.npy')
            # print(accepted, rejected)
            plt.figure()
            for x in accepted:
                plt.scatter(x[0],x[1],color='g')
            for x in rejected:
                plt.scatter(x[0],x[1],color='r')
                
    def plot_last_step(self):
        ''' Determines the last step of exploration (0 if nothing was already computed)
        and plots the results after that step'''
        
        step_data_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)
        
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            self.plot_step(0)
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x.npy' where x is the nb of the step
            self.plot_step(laststep)



