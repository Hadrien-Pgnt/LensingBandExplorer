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

from scipy.spatial import ConvexHull
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
                   
                    # if hop==0: #### TEMPORARY
                    #     index_change=0
                    #     sign=-1
                        
                    if index_change==0:#change spin with the right increment
                        spin += sign*incr[0]
                    else: #change inclination with the right increment
                        incl += sign*incr[1]
                    
                    ### Tests if the point was already computed during previous steps (with a numerical tolerance)
                    if step!=0:
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
                
                # alph=0.002
                alph=0
                # alph = 0.9 * alphashape.optimizealpha(accepted[:,:2])
                hull = alphashape.alphashape(accepted[:,:2], alph)
                # hull_pts = hull.exterior.coords.xy
                hull = hull.simplify(0.003, preserve_topology=False).buffer(0.0005, single_sided=True)
                # hull = hull.buffer(0.0005, single_sided=True)
                
                fig, ax = plt.subplots()
                ax.set_xlabel('Spin parameter',fontsize=24 )
                ax.set_ylabel('Inclination (°)',fontsize=24)
                ax.tick_params(which='both', labelsize=16)
                ax.scatter(self.spin_guess, self.incl_guess, color='tab:brown', s=100, marker='*', alpha=1, label=r'Identification of the $n=2$ ring with a critical curve')
                # for x in accepted:
                #     ax.scatter(x[0],x[1],color='g')
                # for x in rejected:
                #     ax.scatter(x[0],x[1],color='r')
                ax.add_patch(PolygonPatch(hull, fc=blue, ec='k', alpha=0.1, label=r'$n=2$ lensing bands accepting a phoval with the given diameters'))
                ax.legend()
                ax.set_title(r'Spin \& inclination from $n=2$ ring with $(d_+/M,d_-/M)_{\rm measured}$ = (%5.3f, %5.3f)' %(self.dplus, self.dminus), fontsize=18)
                
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

    def plot_step_in_dpm_plane(self, step, fancy = False):
        ''' Plots the results of the random walk after the given step in the (d+, d-) plane'''
        
        if step==0:
            print('No accepted/rejected values computed yet')
        else:
            data_load_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step)
            accepted = np.load(data_load_path+'_accepted.npy')
            rejected = np.load(data_load_path+'_rejected.npy')
            
            plt.figure()
            plt.tick_params(which='both', labelsize=16)
            plt.xlabel(r'$d_+/M$',fontsize=24)
            plt.ylabel(r'$d_-/M$',fontsize=24)
            
            if not(fancy):
                for x in accepted:
                    plt.scatter(*cc.crit_curve_diams(*x[:2]),color='g')
                for x in rejected:
                    plt.scatter(*cc.crit_curve_diams(*x[:2]),color='r')
            else:
                pts = []
                for x in accepted:
                    pts.append(cc.crit_curve_diams(*x[:2]))
                hull = ConvexHull(pts)
                pts = np.array(pts)
                plt.fill(pts[hull.vertices,0], pts[hull.vertices,1], facecolor=(*blue,0.1), edgecolor='k', label = r'Crit. curves with $n=2$ lensing bands accepting a phoval with the given diameters')
    
    def plot_last_step_in_dpm_plane(self, fancy=False):
        ''' Determines the last step of exploration (0 if nothing was already computed)
        and plots the results after that step in the (d+, d-) plane'''
        
        step_data_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)
        
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            self.plot_step_in_dpm_plane(0, fancy)
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x.npy' where x is the nb of the step
            self.plot_step_in_dpm_plane(laststep, fancy)
            
    def is_in_astro_box(self, spin, incl, smin, smax, deltamax):
        ''' Determines whether the measured (d+, d-) is in the parametrized bow with params smin, smax, deltamax
        between the two LB edges for the given spin & incl'''
        
        lb = LensingBand(spin, incl, self.order, self.NN)
        lb.compute_edges_points()
        lb.compute_edges_polar()
        lb.phoval_fit_edges(100)
        dplus_inner, dminus_inner = lb.diams_inner()
        dplus_outer, dminus_outer = lb.diams_outer()
        
        def segment_lensingband(s):
            return [dplus_inner+s*(dplus_outer-dplus_inner), dminus_inner+s*(dminus_outer-dminus_inner)]
        
        smeas = minimize(lambda s: (self.dplus-segment_lensingband(s)[0])**2 + (self.dminus-segment_lensingband(s)[1])**2, x0=[0.5], bounds=[(0,1)])
        deltameas = np.sqrt((self.dplus-segment_lensingband(smeas.x[0])[0])**2 + (self.dminus-segment_lensingband(smeas.x[0])[1])**2)
        # print(smeas.x[0], deltameas, segment_lensingband(smeas.x[0]))
       
        return (smin <= smeas.x[0] <= smax and deltameas <= deltamax)
            
    def compute_subset_astro(self, smin, smax, deltamax):
        '''Determines the subset of spin & inclination within the already computed points
        which are astrophysically plausible
        i.e. when the measured values (d+,d-) are in the parametrized box (with params smin, smax, deltamax)'''
        
        #### Loads last step data
        step_data_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            print('No accepted/rejected values computed yet')
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x.npy' where x is the nb of the step
            data_load_path = step_data_path+'/step_'+str(laststep)
            accepted = np.load(data_load_path+'_accepted.npy')


        #### Tests, for each value in accepted, if the measured pt is in the parametrized box
            astro_accepted = []
            for i in range(len(accepted)):
                print('Processing ' + str(i)+'/'+str(len(accepted)))

                if self.is_in_astro_box(accepted[i][0], accepted[i][1], smin, smax, deltamax):
                    astro_accepted.append(accepted[i])
            
            if not os.path.exists(step_data_path+'_astro'):
                os.makedirs(step_data_path+'_astro')
            
            data_save_path = step_data_path+'_astro/step_'+str(laststep)+'_accepted_smin'+str(smin)+'smax'+str(smax)+'deltamax'+str(deltamax)+'.npy'
            np.save(data_save_path, astro_accepted, allow_pickle=True)
        
                
    def plot_subset_astro(self, smin, smax, deltamax):
        ''' Plots the results of compute_subset_astro'''
        
        #### Loads last step data
        step_data_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            print('No accepted/rejected values computed yet')
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x.npy' where x is the nb of the step
            data_load_path = step_data_path+'/step_'+str(laststep)
            accepted = np.load(data_load_path+'_accepted.npy')
            rejected = np.load(data_load_path+'_rejected.npy')
            astro_accepted = np.load(step_data_path+'_astro/step_'+str(laststep)+'_accepted_smin'+str(smin)+'smax'+str(smax)+'deltamax'+str(deltamax)+'.npy')

            plt.figure()
            plt.xlabel('Spin parameter')
            plt.ylabel('Inclination (°)')
            for x in accepted:
                plt.scatter(x[0],x[1],color='g')
            for x in rejected:
                plt.scatter(x[0],x[1],color='r')
            for x in astro_accepted:
                plt.scatter(x[0],x[1],color='y')
    
    def compute_subregion_astro(self, bounds, Ngrid, smin, smax, deltamax):
        '''Determines the subregion of spin & inclination which are astrophysically plausible
        i.e. when the measured values (d+,d-) are in the parametrized box
        by testing on a Ngrid[0] x Ngrid[1] grid in (a,i) plane 
        with min (resp. max) spin bounds[0] (resp. bounds[1])
        and min (resp. max) inclination bounds[2] (resp. bounds[3])'''

        spingrid = np.linspace(bounds[0], bounds[1], Ngrid[0])
        inclgrid = np.linspace(bounds[2], bounds[3], Ngrid[1])
        
        astro_accepted = []
        count=0
        for spin in spingrid:
            for incl in inclgrid:
                count+=1
                print('Processing '+str(count) +'/'+ str(Ngrid[0]*Ngrid[1])+ ': (a,i)='+str(spin)+', '+str(incl))
                if self.is_in_astro_box(spin, incl, smin, smax, deltamax):
                    astro_accepted.append([spin, incl])
        
        save_folder_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)+'_astro'
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        data_save_path = save_folder_path+'/accepted_smin'+str(smin)+'smax'+str(smax)+'deltamax'+str(deltamax)+'.npy'
        
        ## Aggregates with previous data if existing
        if not os.path.exists(data_save_path):
            np.save(data_save_path, astro_accepted, allow_pickle=True)
        else:
            prev_data = np.load(data_save_path)
            np.save(data_save_path, np.concatenate((prev_data, np.array(astro_accepted))), allow_pickle=True)
    
    def plot_subregion_astro(self, smin, smax, deltamax, fancy=False):
        ''' Plots the results of compute_subregion_astro'''
        
        save_folder_path = self.step_data_dir + '/dplus'+str(self.dplus)+'dminus'+str(self.dminus)+'order'+str(self.order)+'NN'+str(self.NN)+'_astro'
        data_save_path = save_folder_path+'/accepted_smin'+str(smin)+'smax'+str(smax)+'deltamax'+str(deltamax)+'.npy'
        
        if not os.path.exists(data_save_path):
            print('Please compute first')
        else:
            astro_accepted = np.load(data_save_path)
            
            if not(fancy):
                self.plot_last_step()
                for x in astro_accepted:
                    plt.scatter(x[0],x[1],color='y') 
            else:
                self.plot_last_step(fancy=True)
                hull = ConvexHull(astro_accepted)
                plt.fill(astro_accepted[hull.vertices,0], astro_accepted[hull.vertices,1], facecolor=(*green,0.1), edgecolor=(0,0,0,0.1), label = r'$n=2$ lensing bands containing the given diameters in the box with params $s_{\rm min}$=%5.2f, $s_{\rm max}$=%5.2f, $\delta_{\rm max}$=%5.0f $\times 10^{-3}$'%(smin, smax, deltamax*1e3))
                plt.legend(fontsize=15, loc='lower right')

                
                
            

