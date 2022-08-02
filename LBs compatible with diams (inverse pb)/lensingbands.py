#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:41:48 2021

Author: Hadrien Paugnat

For a a fixed lensing band (LB), this code allows to compute
the distance of a point or a phoval to this LB 
(with or without using the phoval fit for the edges)
"""

import numpy as np
import lensingbands_aart as aart
import math
from scipy import interpolate
from scipy.optimize import minimize
from scipy import optimize
import os

########################

def phoval_points(phi, phi0, R0, R1, R2, chi, X):
    ''' Returns the (alpha, beta) points for the (rotated) phoval of params (phi0, R0, R1, R2, chi, X)
    /!\ phi does not correspond to the polar coordinate (sigma), but rather to some intrinsic parametrization of the phoval'''
    f = ( R0 + np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) + (X -chi)*np.cos(phi-phi0) + np.arcsin(chi*np.cos(phi-phi0)))
    fprime = (R1**2-R2**2)*np.sin(phi-phi0)*np.cos(phi-phi0)/np.sqrt(R1**2 * np.sin(phi-phi0)**2 + R2**2 * np.cos(phi-phi0)**2) - (X -chi)*np.sin(phi-phi0) - chi*np.sin(phi-phi0)/np.sqrt(1-(chi*np.cos(phi-phi0))**2)
    x = f*np.cos(phi) - fprime *np.sin(phi)
    y = f*np.sin(phi) + fprime *np.cos(phi)
    return x+y*1j

########################

class LensingBand:
    ''' To fully describe a lensing band for a fixed spin, inclination and order'''
    
    def __init__(self, spin, incl, order, NN):
        self.spin = spin #spin of the BH
        self.incl = incl # observer inclination
        self.order = order # order of the lensing band (mbar)
        
        self.NN = NN #nb of points computed for the edges        
        self.inner_edge = np.zeros([2*NN, 2])
        self.outer_edge = np.zeros([2*NN, 2])
        
        address = os.path.abspath(__file__).replace('\\','/')
        
        ##Creates a directory 'Edge_data' if not already existing
        self.edge_data_dir = '/'.join(address.split("/")[:-1]) + '/Edge_data'
        if not os.path.exists(self.edge_data_dir):
            os.makedirs(self.edge_data_dir)
                  
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

    def compute_edges_points(self):
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
    
    def phoval_fit_edges(self, Ncheck):
        ''' Fits the edges to a (rotated) phoval, using Ncheck points and saves the phoval params'''
        
        phi_check = np.linspace(-np.pi,np.pi,Ncheck)
        
        def RMSdistance_phoval_outer(params):
            try:
                z = phoval_points(phi_check,*params)
                sigma = np.angle(z) #polar angles of the phoval points
                radius = np.vectorize(self.radius_outer)(sigma) #corresponding radii for the outer edge of the lensing band
                return np.sqrt(np.sum(np.abs(z-radius*np.exp(1j*sigma))**2))
            except ValueError:
                return np.inf

        self.phoval_outer = minimize(RMSdistance_phoval_outer, x0=([0.,5.,5.,4.,0.,1.]), bounds=[(-math.pi,math.pi),(0,np.inf),(0,np.inf),(0,np.inf),(-1,1),(-np.inf,np.inf)])
        
        def RMSdistance_phoval_inner(params):
            try:
                z = phoval_points(phi_check,*params)
                sigma = np.angle(z) #polar angles of the phoval points
                radius = np.vectorize(self.radius_inner)(sigma) #corresponding radii for the inner edge of the lensing band
                return np.sqrt(np.sum(np.abs(z-radius*np.exp(1j*sigma))**2))
            except ValueError:
                return np.inf

        self.phoval_inner = minimize(RMSdistance_phoval_inner, x0=([0.,5.,5.,4.,0.,1.]), bounds=[(-math.pi,math.pi),(0,np.inf),(0,np.inf),(0,np.inf),(-1,1),(-np.inf,np.inf)])
        
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
        '''Determines the distance of a phoval of parameters params to the lensing band'''
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
    
    def replace_edges_by_phoval_fits(self):
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
    
    def diams_inner(self):
        '''Returns (d+, d-) for the phoval fit of the inner edge'''
        p = self.phoval_inner.x
        return (2*(p[1]+ max(p[2], p[3])), 2*(p[1]+ min(p[2], p[3])))
    
    def diams_outer(self):
        '''Returns (d+, d-) for the phoval fit of the outer edge'''
        p = self.phoval_outer.x
        return (2*(p[1]+ max(p[2], p[3])), 2*(p[1]+ min(p[2], p[3])))
           
