#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:41:48 2021

Author: Hadrien Paugnat

This code computes the lensings bands (LBs) compatible with a fixed point in the (d+,d-) plane, assumed to be a phoval
"""

from lensingbands import LensingBand
import crit_curve as cc

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, root
from scipy import interpolate
import random as rnd
import os

# from scipy.spatial import ConvexHull
import alphashape
from descartes import PolygonPatch

## For nicer plots

from matplotlib import rcParams
rcParams['font.family']='serif'
rcParams['text.usetex']=True 

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
        
        self.firstguess_phoval_params = None #rotated phoval parameters   
        
        self.spin_guess = spinguess
        self.incl_guess = inclguess
        
        self.address = os.path.abspath(__file__).replace('\\','/')
            
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
        sol = root(to_solve, [self.spin_guess,self.incl_guess], tol=1e-10)
        
        self.spin_guess = sol.x[0]
        self.incl_guess = sol.x[1]
        
        ''' Then fits the obtained crit curve to a phoval (with fixed d+,d-) '''
        
        self.firstguess_phoval_params = cc.crit_curve_phoval_fit_with_diam_constraint(self.spin_guess, self.incl_guess, self.dplus, self.dminus)

        
    def explore_one_step(self, incr, startpoint, nhops, tol, Ncheck):
        ''' Determines the last step of exploration (0 if nothing was already computed)
        then launches explore_at_step at next step with the given parameters for the random walk  '''
        
        step_data_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)
        
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            os.makedirs(step_data_path)
            self.explore_at_step(0, incr, startpoint, nhops, tol, Ncheck)
        
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x_accepted.npy' or 'step_x_rejected.npy' where x is the nb of the step
            self.explore_at_step(laststep, incr, startpoint, nhops, tol, Ncheck)

        
    def explore_at_step(self, step, incr, startpoint, nhops, tol, Ncheck):
        '''Loads the values for accepted/rejected pts in the (a,i) plane previously aggregated during *step* steps
        Performs a new step (the *step+1*): a random walk starting from *startpoint*, 
        walking *nhops* times on a grid with a *incr[0]* increment for spin and a *incr[1]* increment for inclination
        A point of the (a,i) plane is accepted if the corresponding lensing band 
        contains a phoval with the "measured" max/min diameters self.dplus and self.dminus
        (i.e. this phoval has distance to the LB = 0, with a tolerance *tol* to account for numerical artifacts)'''
      
        data_load_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step)
        data_save_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step+1)
        
        if step!=0:
            #Loads the accepted/rejected values from the previous step
            accepted_base = np.load(data_load_path+'_accepted.npy')
            rejected_base = np.load(data_load_path+'_rejected.npy')

            if startpoint=='critical curve guess':
                # spin & incl guess + params of the phoval best fitting the corresponding critical curve (within diameter constraints) 
                spin, incl = self.spin_guess, self.incl_guess
                params = self.firstguess_phoval_params
            else:
                # Startpoint should be specified in the form of a string with spin and inclination separated by a ';'
                # (ex: '0.44;57.2')
                # Program will pick the accepted value closest to this point
                
                start_spin = float(startpoint.split(";")[0])
                start_incl = float(startpoint.split(";")[1])
                
                index=np.argmin(np.abs(start_spin-accepted_base[:,0])+np.abs(start_incl-accepted_base[:,1])) #index in accepted_base of the accepted (a,i) point closest to the coords given in startpoint
                spin, incl = accepted_base[index,0], accepted_base[index,1]
                params=np.copy(accepted_base[index, 2:]) # params of the corresponding accepted rotated phoval 
        
            rejected = list(rejected_base)
            accepted = list(accepted_base)
        
        else: #if step==0 (i.e. if nothing has already been computed)
            spin, incl = self.spin_guess, self.incl_guess
            params = self.firstguess_phoval_params
            # guess from crit curve identification by default
            accepted = []
            rejected = []


        ## Random walk

        for hop in range(nhops):
            print(hop)
            
            prev_params = np.copy(params)
            found_accepted = False
            already_tried =[]
            
            while not(found_accepted) and len(already_tried)<4:
                index_change = rnd.randint(0,1) #randomly pick a or i
                sign = 2*rnd.randint(0,1) - 1 # randomly pick +/-1   
              
                if (index_change,sign) not in already_tried:
                    if index_change==0:#change spin with the right increment
                        spin += sign*incr[0]
                    else: #change inclination with the right increment
                        incl += sign*incr[1]
                    
                    ### Tests if the point was already computed during previous steps (with a numerical tolerance)
                    index_a = np.argmin(np.abs(spin-accepted_base[:,0])+np.abs(incl-accepted_base[:,1]))
                    index_r = np.argmin(np.abs(spin-rejected_base[:,0])+np.abs(incl-rejected_base[:,1]))
                    if step!=0 and np.abs(spin-accepted_base[index_a,0])+np.abs(incl-accepted_base[index_a,1]) <= 1e-11:
                        #point already accepted
                        params = accepted_base[index_a,2:]
                        found_accepted = True
                    elif step!=0 and np.abs(spin-rejected_base[index_r,0])+np.abs(incl-rejected_base[index_r,1]) <= 1e-11:
                        #point already rejected
                        params=prev_params
                        already_tried.append((index_change,sign))
                    elif spin<=0. or spin >=1. or incl<=0. or incl >= 90.:
                        #point rejected because spin or incl is not in the right range
                        rejected.append([spin,incl,*params])
                        params=prev_params
                        already_tried.append((index_change,sign))
                        
                    else:
                            
                        ## Instantiation of a LB, computation of its edges and its phoval fits
                        lb = LensingBand(spin, incl, self.order, self.NN)
                        lb.compute_edges_points()
                        lb.compute_edges_polar()
                        lb.phoval_fit_edges(100)
                        lb.replace_edges_by_phoval_fits()
                            
                        if lb.dist_phoval_to_band(params,Ncheck)<= tol:
                            accepted.append([spin, incl,*params])
                            found_accepted = True
                        
                        else:
                            # ## minimize over phi0, chi and X the distance to the lensing band (approximative)
                            # best_add_params = minimize(lambda p: lb.dist_phoval_to_band([p[0],*params[1:4],p[1],p[2]],Ncheck), x0=([params[0],*params[4:]]),bounds=[(-math.pi,math.pi),(-1,1),(-np.inf,np.inf)])
                            # bestparams = [best_add_params.x[0], *params[1:4], *best_add_params.x[1:]]
                            
                            ## minimize over phi0, R0, chi and X (R1 and R2 are given by keeping d+, d- constant) the distance to the lensing band
                            best_add_params = minimize(lambda p: lb.dist_phoval_to_band([p[0],p[1],0.5*self.dplus-p[1],0.5*self.dminus-p[1],p[2],p[3]],Ncheck), x0=([*params[0:2],*params[4:]]),bounds=[(-math.pi,math.pi),(0., self.dminus),(-1,1),(-np.inf,np.inf)])
                            bestparams = [*best_add_params.x[0:2], 0.5*self.dplus-best_add_params.x[1],0.5*self.dminus-best_add_params.x[1], *best_add_params.x[2:]]
    
                            if lb.dist_phoval_to_band(bestparams, Ncheck) <= tol:
                                params = bestparams
                                accepted.append([spin,incl,*params])
                                found_accepted = True
                                
                            else:                   
                                rejected.append([spin,incl,*params])
                                params=prev_params
                                already_tried.append((index_change,sign))
                            
        np.save(data_save_path+'_accepted.npy', accepted, allow_pickle=True)
        np.save(data_save_path+'_rejected.npy', rejected, allow_pickle=True)
     
    def plot_step(self, step, fancy=False):
        ''' Plots the results of the random walk after the given step'''
        
        if step==0:
            print('No accepted/rejected values computed yet')
        else:
            data_load_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step)
            accepted = np.load(data_load_path+'_accepted.npy')
            rejected = np.load(data_load_path+'_rejected.npy')
            
            if not(fancy):

                plt.figure()
                plt.xlabel('Spin parameter')
                plt.ylabel('Inclination (°)')
                for x in accepted:
                    plt.scatter(x[0],x[1],color='g')
                for x in rejected:
                    plt.scatter(x[0],x[1],color='r')
            else:
                
                alph = alphashape.optimizealpha(accepted[:,:2])
                hull = alphashape.alphashape(accepted[:,:2], alph)
                hull_pts = hull.exterior.coords.xy
                
                fig, ax = plt.subplots()
                ax.set_xlabel('Spin parameter')
                ax.set_ylabel('Inclination (°)')
                ax.scatter(self.spin_guess, self.incl_guess, color='tab:brown', s=100, marker='*', alpha=1, label=r'Identification of the $n=2$ ring with a critical curve')
                ax.scatter(hull_pts[0], hull_pts[1], color=green)
                ax.add_patch(PolygonPatch(hull, fc=blue, ec='k', alpha=0.1, label=r'$n=2$ lensing bands accepting a phoval with the given diameters'))
                ax.legend()
                
    def plot_last_step(self, fancy=False):
        ''' Determines the last step of exploration (0 if nothing was already computed)
        and plots the results after that step'''
        
        step_data_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)
        
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            self.plot_step(0, fancy)
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x.npy' where x is the nb of the step
            self.plot_step(laststep, fancy)



