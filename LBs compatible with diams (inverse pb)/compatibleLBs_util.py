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
# spinguess, inclguess = 0.9, 20

dm = DiameterMeasurement(dplus_measured, dminus_measured, order, NN, spinguess, inclguess )


### Uses the crit. curve map (a,i)->(d+,d-) to get a precise starting point for the spin & incl
### then fits the corresponding crit. curve map to a phoval (with fixed d+,d-)

dm.guess_spin_incl_from_crit_curve()
# print(dm.spin_guess, dm.incl_guess)
# pcrit_constrained = cc.crit_curve_phoval_fit_with_diam_constraint(dm.spin_guess, dm.incl_guess, 10.318483143214817, 10.235834607023905,plot=True)
# print(dm.firstguess_phoval_params)


### Random walk parameters

# incr = (0.01, 2.5) #increments for spin and inclination, respectively
incr = (0.002, 1.25)
startpoint = '0.596;42.2'
nhops = 50
tol = 5e-9 
Ncheck = 100

# dm.explore_one_step(incr, startpoint, nhops, tol, Ncheck)
# dm.plot_last_step(fancy=False)
# dm.plot_last_step(fancy=True)
# dm.plot_last_step_in_dpm_plane(fancy = False)
    

### Several random walks

# for startpoint in ['0.554;22.', '0.554;24.5', '0.578; 52.25', '0.598;31.', '0.47; 44.6']:
#     print(startpoint)
#     dm.explore_one_step(incr, startpoint, nhops, tol, Ncheck)
#     dm.plot_last_step(fancy=False)



### Astrophysically plausible subset

smin = 0.18
smax = 0.75
deltamax = 5e-3

## Gives the already accepted points which are also contains the point in their parametrized box
# dm.compute_subset_astro(smin, smax, deltamax)
# dm.plot_subset_astro(smin, smax, deltamax)

## More precise determination of the subregion in the (a,i) plane : use a grid between the given limits

bounds = [0.485, 0.575, 32, 50]
Ngrid = [15, 20]

# bounds = [0.56857143, 0.574, 33, 36]
# Ngrid = [5, 5]

# dm.compute_subregion_astro(bounds, Ngrid, smin, smax, deltamax)
dm.plot_subregion_astro(smin, smax, deltamax, fancy=True)



