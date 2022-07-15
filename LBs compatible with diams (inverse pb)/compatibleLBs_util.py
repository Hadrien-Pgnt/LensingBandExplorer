# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:06:50 2022

Author: Hadrien Paugnat

This code computes the lensings bands (LBs) compatible with a fixed point in the (d+,d-) plane, assumed to be a phoval
"""
from compatibleLBs_randomwalk import DiameterMeasurement
import crit_curve as cc


# ### Gives the values (d+, d-) for a Kerr critical curve of given spin & inclination
# print(cc.crit_curve_diams(0.5,45)) 

# ### Plots a map (a,i)->(d+,d-) for the critical curves on a grid of size n_grid x n_grid 
# ### (and computes it if not already precomputed with that size)
# n_grid = 200
# grid = cc.CritCurveGrid(n_grid)
# grid.plot_map()


### Instantiation

# these values of d+, d- correspond to the critical curve for a=0.5, i=45°
dplus_measured, dminus_measured = 10.318483143214817, 10.235834607023905

# # these values of d+, d- correspond to the critical curve for a=0.94, i=17°
# dplus_measured, dminus_measured = 9.833778588299497, 9.7317131865179

order = 2 #order of the image (here we are interested in the n=2 ring)
NN = 1000 #nb of points computed for the LB edges 

# eyeball estimates for the spin & inclination (use the crit. curve map (a,i)->(d+,d-))
spinguess, inclguess = 0.45, 50
spinguess, inclguess = 0.9, 20

dm = DiameterMeasurement(dplus_measured, dminus_measured, order, NN, spinguess, inclguess )


### Uses the crit. curve map (a,i)->(d+,d-) to get a precise starting point for the spin & incl
### then fits the corresponding crit. curve map to a phoval (with fixed d+,d-)
dm.guess_spin_incl_from_crit_curve()
# print(dm.spin_guess, dm.incl_guess)
pcrit_constrained = cc.crit_curve_phoval_fit_with_diam_constraint(dm.spin_guess, dm.incl_guess, 10.318483143214817, 10.235834607023905,plot=True)
print(dm.phoval_params)
