#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:41:48 2021

Author: Hadrien Paugnat

This code computes the phovals allowed in a fixed lensing band (LB) - fully 
determined by spin, inclination and order of the LB
"""

import lensingbands_aart as aart

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path as paths
import math
from scipy import interpolate
from scipy.optimize import minimize
from scipy import optimize
from scipy.spatial import ConvexHull
import random as rnd
import os

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

def point_in_hull(point, hull):
    """
    Check if points in p are inside the concave hull
    """
    concave=paths.Path(hull)
    return concave.contains_points(point)

def phoval_points(phi, phi0, R0, R1, R2, chi, X):
    ''' Returns the (alpha, beta) points for the (rotated) phoval of params (phi0, R0, R1, R2, chi, X)
    /!\ phi does not correspond to the polar coordinate (sigma), but rather to some intrinsic parametrization of the phoval'''
    f = ( R0 + np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) + (X -chi)*np.cos(phi-phi0) + np.arcsin(chi*np.cos(phi-phi0)))
    fprime = (R1**2-R2**2)*np.sin(phi-phi0)*np.cos(phi-phi0)/np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) - (X -chi)*np.sin(phi-phi0) - chi*np.sin(phi-phi0)/np.sqrt(1-(chi*np.cos(phi-phi0))**2)
    x = f*np.cos(phi) - fprime *np.sin(phi)
    y = f*np.sin(phi) + fprime *np.cos(phi)
    return x+y*1j

n_points = int(1e5)
def crit_curve_diams(a,incl):   
    ''' Gives the max/min diameters of the Kerr critical curve for spin a and inclination incl'''
    
    thetaobs = incl*math.pi/180  #in radians
    
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

    d_ortho= np.max(x) - np.min(x) #min diameter is orthogonal to the spin axis, i.e. for phi_p=0°
    d_parallel= 2*np.max(y)  # max diameter is parallel to the spin axis, i.e. for phi_p=90°
    #we only plotted the half-circle here, but the figure is symmetric along the x axis  
    
    return (d_parallel, d_ortho)


########################

class LensingBand:
    ''' To explore a lensing band for a fixed spin, inclination and order
    and compute the phovals allowed within that band'''
    def __init__(self, spin, incl, order, NN):
        self.spin = spin #spin of the BH
        self.incl = incl # observer inclination
        self.order = order # order of the lensing band (mbar)
        
        self.NN = NN #nb of points computed for the edges        
        self.inner_edge = np.zeros([2*NN, 2])
        self.outer_edge = np.zeros([2*NN, 2])
        
        self.address = os.path.abspath(__file__).replace('\\','/')
        
        ##Creates a directory 'Edge_data' if not already existing
        self.edge_data_dir = '/'.join(self.address.split("/")[:-1]) + '/Edge_data'
        if not os.path.exists(self.edge_data_dir):
            os.makedirs(self.edge_data_dir)
            
        ##Creates a directory 'Step_data' if not already existing
        self.step_data_dir = '/'.join(self.address.split("/")[:-1]) + '/Steps_data'
        if not os.path.exists(self.step_data_dir):
            os.makedirs(self.step_data_dir)
        
        #Initialization of some attributes used later on for interpolation
        self.minangleinner = 0
        self.minangleouter = 0
        self.maxangleinner = 0
        self.maxangleouter = 0
        self.router_bound = None
        self.rinner_bound = None
        self.router_missing = None
        self.rinner_missing = None
        
        # Best-fit (rotated) phoval parameters for the inner and outer edges (initialized)
        self.phoval_inner = None
        self.phoval_outer = None

    def compute_edges_points(self, plot=False):
        ''' Loads the lensing band outer/inner edges if already computed
        otherwise computes them using AART (2*self.NN points will be computed for each edge)
        Plots the lensing band if plot is True'''
        
        lb_data_path = self.edge_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)+'.npy'
        
        #Loads lensing band edge data if already existing
        if os.path.exists(lb_data_path):
            self.inner_edge = np.load(lb_data_path, allow_pickle=True)[0]
            self.outer_edge = np.load(lb_data_path, allow_pickle=True)[1]
        
        #Else computes it and saves the result in Edge_data repository
        else:
            data=aart.Shadow(self.spin,self.incl)
            alpha, beta = aart.spacedmarks(data[0], data[1], self.NN)
            data=(np.append(alpha,alpha), np.append(beta,-beta))
            
            for i in range(2*self.NN):
                m1=optimize.root(aart.nlayers, 1.0001, args=(self.spin,self.incl,90,data[0][i],data[1][i],self.order))
                m2=optimize.root(aart.nlayers, 0.9999, args=(self.spin,self.incl,90,data[0][i],data[1][i],self.order))
                
                self.outer_edge[i]=1.0001*m1.x[0]*np.array([data[0][i],data[1][i]])
                self.inner_edge[i]=0.9999*m2.x[0]*np.array([data[0][i],data[1][i]])
                
            np.save(lb_data_path, np.array((self.inner_edge, self.outer_edge)))
        
        if plot:
            plt.scatter(self.inner_edge[:,0], self.inner_edge[:,1], color = orange, s=1)
            plt.scatter(self.outer_edge[:,0], self.outer_edge[:,1], color='c', s=1)

    
    # def is_in_band(self,point):
    #     ''' Tests whether point is in the lensing band (represented by the concave hull of its edges) 
    #     Returns "bullseye" if in the band, "outer" if outside of the outer edge, "inner" if inside the inner edge '''
        
    #     concave_inner = paths.Path(self.inner_edge)
    #     concave_outer = paths.Path(self.outer_edge)
        
    #     if concave_inner.contains_point(point):
    #         return "inner"
    #     else:
    #         if not(concave_outer.contains_point(point)):
    #             return "outer"
    #         else:
    #             return "bullseye"
    
    def compute_edges_polar(self):
        ''' Interplates between the results of compute_edges_points to get a polar equaation r(phi) for both edges '''
        
        outerring = self.outer_edge[:,0] + 1j*self.outer_edge[:,1]
        innerring = self.inner_edge[:,0] + 1j*self.inner_edge[:,1] 
        
        # Minimal and maximal angular value in ]-pi,pi] between which we can directly interpolate using the points
        self.minangleinner = np.min(np.angle(innerring))
        self.minangleouter = np.min(np.angle(outerring))
        self.maxangleinner = np.max(np.angle(innerring))
        self.maxangleouter = np.max(np.angle(outerring))
        
        # Interpolation between these extremal angles        
        self.router_bound = interpolate.interp1d(np.angle(outerring), np.abs(outerring), kind='cubic')
        self.rinner_bound = interpolate.interp1d(np.angle(innerring), np.abs(innerring), kind='cubic')
        
        # Interpolation outside of these angles, using sigma -> sigma + 2*pi invariance
        self.router_missing = interpolate.interp1d(np.angle(outerring)+2*np.pi*(np.angle(outerring)<0), np.abs(outerring), kind='cubic') 
        self.rinner_missing = interpolate.interp1d(np.angle(innerring)+2*np.pi*(np.angle(innerring)<0), np.abs(innerring), kind='cubic')

    def radius_outer(self, sig):
        ''' Polar equation for the outer edge of the lensing band, covering all polar angles '''
        sigma=np.angle(np.exp(1j*sig)) #value in ]-pi,pi]
        if self.minangleouter <= sigma <= self.maxangleouter:
            return self.router_bound(sigma)
        elif self.maxangleouter < sigma :
            return self.router_missing(sigma)
        elif self.minangleouter > sigma:
            return self.router_missing(sigma+2*math.pi)
        else:
            raise ValueError
            
    def radius_inner(self, sig):
        ''' Polar equation for the inner edge of the lensing band '''
        sigma=np.angle(np.exp(1j*sig)) #value in ]-pi,pi]
        if self.minangleinner <= sigma <= self.maxangleinner:
            return self.rinner_bound(sigma)
        elif self.maxangleinner < sigma :
            return self.rinner_missing(sigma)
        elif self.minangleinner > sigma:
            return self.rinner_missing(sigma+2*math.pi)
        else:
            raise ValueError
            
    def phoval_fit_edges(self, Ncheck, plot = False):
        ''' Fits the edges to a (rotated) phoval, using Ncheck points and saves the phoval params'''
        
        phi_check = np.linspace(-np.pi,np.pi,Ncheck)
        
        def RMSdistance_phoval_outer(params):
            z = phoval_points(phi_check,*params)
            sigma = np.angle(z) #polar angles of the phoval points
            radius = np.vectorize(self.radius_outer)(sigma) #corresponding radii for the outer edge of the lensing band
            return np.sqrt(np.sum(np.abs(z-radius*np.exp(1j*sigma))**2))

        self.phoval_outer = minimize(RMSdistance_phoval_outer, x0=([0.,5.,5.,4.,0.,1.]), bounds=[(-math.pi,math.pi),(0,np.inf),(0,np.inf),(0,np.inf),(-1,1),(-np.inf,np.inf)])
        
        def RMSdistance_phoval_inner(params):
            z = phoval_points(phi_check,*params)
            sigma = np.angle(z) #polar angles of the phoval points
            radius = np.vectorize(self.radius_inner)(sigma) #corresponding radii for the inner edge of the lensing band
            return np.sqrt(np.sum(np.abs(z-radius*np.exp(1j*sigma))**2))

        self.phoval_inner = minimize(RMSdistance_phoval_inner, x0=([0.,5.,5.,4.,0.,1.]), bounds=[(-math.pi,math.pi),(0,np.inf),(0,np.inf),(0,np.inf),(-1,1),(-np.inf,np.inf)])
        
        if plot:
            pts_outer = phoval_points(phi_check,*self.phoval_outer.x)
            pts_inner = phoval_points(phi_check,*self.phoval_inner.x)
            
            plt.scatter(self.inner_edge[:,0], self.inner_edge[:,1], color = orange, s=1, label='Computed points (inner)')
            plt.scatter(self.outer_edge[:,0], self.outer_edge[:,1], color='c', s=1, label='Computed points (outer)')
            plt.scatter(np.real(pts_inner), np.imag(pts_inner), color = red, s=1, label='Phoval fit (inner)')
            plt.scatter(np.real(pts_outer), np.imag(pts_outer), color=blue, s=1, label='Phoval fit (outer)')
    
            plt.legend()
        
    def is_in_band(self,z):
        ''' Tests whether point (represented by a complex number z) is in the lensing band
        Returns "bullseye" if in the band, "outer" if outside of the outer edge, "inner" if inside the inner edge '''
        
        r_inner = self.radius_inner(np.angle(z))
        r_outer = self.radius_outer(np.angle(z))
        
        if np.abs(z) < r_inner:
            return "inner"
        elif np.abs(z) > r_outer:
            return "outer"
        else:
            return "bullseye"
            
    def dist_to_band(self,z):
        '''Determines the distance of a point to the lensing band region'''
        test = self.is_in_band(z)
        
        if test=="bullseye":
            return 0
        elif test=="inner":
            closest = minimize(lambda sig: np.abs(z-self.radius_inner(sig)*np.exp(1j*sig)),x0 = np.angle(z))
            return np.abs(z-self.radius_inner(closest.x)*np.exp(1j*closest.x))[0]
        else:
            closest = minimize(lambda sig: np.abs(z-self.radius_outer(sig)*np.exp(1j*sig)),x0 = np.angle(z))
            return np.abs(z-self.radius_outer(closest.x)*np.exp(1j*closest.x))[0]
        
    def dist_phoval_to_band(self,params, Ncheck):
        '''Determines the distance of a phoval of parameters p to the lensing band'''
        dist=0
        phi_check = np.linspace(-np.pi,np.pi,Ncheck)
        for phi in phi_check:
            try:
                z = phoval_points(phi,*params)
                d=self.dist_to_band(z)
                if d>dist:
                    dist=d
            except ValueError:
                print(z)
        return dist
    
    def replace_edges_by_phoval_fits(self, plot=False):
        '''Replaces the LB edges by its phoval fits (represented by 2*NN points)
        for future computations'''
        phi_check = np.linspace(-np.pi,np.pi,2*self.NN, endpoint=False)
        points_outer = phoval_points(phi_check,*self.phoval_outer.x)
        points_inner = phoval_points(phi_check,*self.phoval_inner.x)
        
        self.outer_edge[:,0] = np.real(points_outer)
        self.outer_edge[:,1] = np.imag(points_outer)
        self.inner_edge[:,0] = np.real(points_inner)
        self.inner_edge[:,1] = np.imag(points_inner)
    
        self.compute_edges_polar()
        
        ''' Plots the LB edges, and scatters the points of the "medium" phoval
        (used by default as the starting point for the random walk)'''
        
        if plot:
            plt.figure()
            sigma = np.linspace(-np.pi,np.pi, 2*self.NN)
            
            rinner = np.vectorize(self.radius_inner)(sigma)
            router = np.vectorize(self.radius_outer)(sigma)
            plt.plot(rinner*np.cos(sigma), rinner*np.sin(sigma), c=orange, label='Phoval fit (inner)' )
            plt.plot(router*np.cos(sigma), router*np.sin(sigma), c='c', label='Phoval fit (outer)' )
            
            params_medium = 0.5*(np.array(self.phoval_outer.x)+np.array(self.phoval_inner.x))
            medium_pts = phoval_points(sigma,*params_medium)
            plt.plot(np.real(medium_pts), np.imag(medium_pts), 'k--', label='Medium phoval')
            plt.legend()
        
           
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
                        # ## minimize over phi0, chi and X the distance to the lensing band (approximative)
                        # best_add_params = minimize(lambda p: self.dist_phoval_to_band([p[0],*params[1:4],p[1],p[2]],Ncheck), x0=([params[0],*params[4:]]),bounds=[(-math.pi,math.pi),(-1,1),(-np.inf,np.inf)])
                        # bestparams = [best_add_params.x[0], *params[1:4], *best_add_params.x[1:]]
                        
                        ## minimize over phi0, R0, chi and X (R1 and R2 are given by keeping d+, d- constant) the distance to the lensing band
                        best_add_params = minimize(lambda p: self.dist_phoval_to_band([p[0],p[1],0.5*d_plus-p[1],0.5*d_minus-p[1],p[2],p[3]],Ncheck), x0=([*params[0:2],*params[4:]]),bounds=[(-math.pi,math.pi),(0., d_minus),(-1,1),(-np.inf,np.inf)])
                        bestparams = [*best_add_params.x[0:2], 0.5*d_plus-best_add_params.x[1],0.5*d_minus-best_add_params.x[1], *best_add_params.x[2:]]


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

    
    def plot_step(self, step, fancy=False):
        ''' Plots the results of the random walk after the given step'''
        
        if step==0:
            print('No accepted/rejected values computed yet')
        else:
            data_load_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)+'/step_'+str(step)
            accepted = np.load(data_load_path+'_accepted.npy')
            rejected = np.load(data_load_path+'_rejected.npy')
            # print(accepted, rejected)
            if not(fancy):
                plt.figure()
                for x in accepted:
                    plt.scatter(x[0],x[1],color='g')
                for x in rejected:
                    plt.scatter(x[0],x[1],color='r')
                    
            else:
                plt.figure(figsize=(17,11))
                line1 = plt.plot([2*(self.phoval_inner.x[1]+self.phoval_inner.x[2])-0.02,2*(self.phoval_outer.x[1]+self.phoval_outer.x[3])+0.02], [2*(self.phoval_inner.x[1]+self.phoval_inner.x[2])-0.02,2*(self.phoval_outer.x[1]+self.phoval_outer.x[3])+0.025], color='tab:grey', linestyle='--', label = r'$d_+=d_-$ curve')
                
                #manually add the LB edges
                accepted = np.concatenate((accepted, [[2*(self.phoval_outer.x[1]+self.phoval_outer.x[2]),2*(self.phoval_outer.x[1]+self.phoval_outer.x[3]),*self.phoval_outer.x]],[[2*(self.phoval_inner.x[1]+self.phoval_inner.x[2]),2*(self.phoval_inner.x[1]+self.phoval_inner.x[3]),*self.phoval_inner.x]]))
                
                hull = ConvexHull(accepted[:,0:2])        
                fill1 = plt.fill(accepted[hull.vertices,0], accepted[hull.vertices,1], facecolor=(*blue,0.1), edgecolor='k', label = r'Allowed phovals in the lensing band')
                
                scatt2 = plt.scatter(2*(self.phoval_outer.x[1]+self.phoval_outer.x[2]),2*(self.phoval_outer.x[1]+self.phoval_outer.x[3]),color='c', s=100, marker='o', label=r'Outer edge of the $n=2$ lensing band')
                scatt3 = plt.scatter(2*(self.phoval_inner.x[1]+self.phoval_inner.x[2]),2*(self.phoval_inner.x[1]+self.phoval_inner.x[3]),color=orange, s=100, marker='o', label=r'Inner edge of the $n=2$ lensing band')

                critcurve = crit_curve_diams(self.spin,self.incl)
                plt.scatter(critcurve[0], critcurve[1], color='tab:brown', s=100, marker='*', alpha=1, label='Critical curve')
                

                # #### TEMPORARY: compare with old method ####
                # allowed=np.load('spin500angle45_allowedvalues_step19.npy')
                # # ## We add the outer and inner edges in the allowed values - with their phoval best fit
                # allowed = np.concatenate((allowed,  [[2*(self.phoval_outer.x[1]+self.phoval_outer.x[2]),2*(self.phoval_outer.x[1]+self.phoval_outer.x[3]),*self.phoval_outer.x]],[[2*(self.phoval_inner.x[1]+self.phoval_inner.x[2]),2*(self.phoval_inner.x[1]+self.phoval_inner.x[3]),*self.phoval_inner.x]]))

                # hull = ConvexHull(allowed[:,0:2])        
                # plt.fill(allowed[hull.vertices,0], allowed[hull.vertices,1], facecolor=(*green,0.1), edgecolor='k', label = r'Allowed phovals in the $n=2$ lensing band (old method)')
                
                ###################
                
                plt.legend(fontsize=12, loc = 'upper left')
                plt.tick_params(which='both', labelsize=16)
                plt.xlabel(r'$d_+/M$',fontsize=24)
                plt.ylabel(r'$d_-/M$',fontsize=24)


    def plot_last_step(self, fancy=False):
        ''' Determines the last step of exploration (0 if nothing was already computed)
        and plots the results after that step'''
        
        step_data_path = self.step_data_dir + '/spin'+str(self.spin)+'incl'+str(self.incl)+'order'+str(self.order)+'NN'+str(self.NN)
        
        # If a folder does not already exist, nothing was computed so "last step" is 0 (and we create the folder) 
        if not os.path.exists(step_data_path):
            self.plot_step(0,fancy)
        else:
            laststep = max([int(file[5:-13]) for file in os.listdir(step_data_path)]) #the file names should be of the form 'step_x.npy' where x is the nb of the step
            self.plot_step(laststep, fancy)




        






