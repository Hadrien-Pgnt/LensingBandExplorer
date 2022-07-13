# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:06:50 2022

Author: Hadrien Paugnat

This code computes the lensings bands (LBs) compatible with a fixed point in the (d+,d-) plane, assumed to be a phoval
"""
from compatibleLBs_randomwalk import DiameterMeasurement
import crit_curve as cc

# Gives the values (d+, d-) for a Kerr critical curve of given spin & inclination
print(cc.crit_curve(0.5,45)) 

# Plots a map (a,i)->(d+,d-) for the critical curves on a grid of size n_grid x n_grid 
# (and computes it if not already precomputed with that size)
n_grid = 200
grid = cc.CritCurveGrid(n_grid)
grid.plot_map()



dm = DiameterMeasurement(10.318483143214817, 10.235834607023905, 2, 1000, 0.45, 50)
# these values of d+, d- correspond to the critical curve for a=0.5, i=45°

# dm = DiameterMeasurement(9.833778588299497, 9.7317131865179, 2, 1000, 0.9, 20)
# # these values of d+, d- correspond to the critical curve for a=0.94, i=17°

dm.guess_spin_incl_from_crit_curve()
print(dm.spin_guess, dm.incl_guess)


print(cc.crit_curve(dm.spin_guess, dm.incl_guess))

pcrit = cc.crit_curve_phoval_fit(dm.spin_guess, dm.incl_guess, plot=True)
print(2*(pcrit[0]+pcrit[1]), 2*(pcrit[0]+pcrit[2]))

