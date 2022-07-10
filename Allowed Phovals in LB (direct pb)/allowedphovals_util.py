# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:41:48 2021

Author: Hadrien Paugnat

This code computes the phovals allowed in a fixed lensing band (LB) - fully 
determined by spin, inclination and order of the LB
"""

from allowedphovals_randomwalk import LensingBand
import numpy as np
from matplotlib import path

##############

### LB parameters
spin = 0.5
incl = 45
order = 2
NN = 100 #nb of points computed for the edges

### Instantiation and calculation of the LB edges (+ their phoval fits)
lb = LensingBand(spin, incl, order, NN)
lb.compute_edges_points(plot=False)
lb.compute_edges_polar()
lb.phoval_fit_edges(100, plot = False)

### NB: there is a (small) difference between the LB edges and their phoval fits
### To explore the allowed phovals, we work with the phoval fit of the edges 
lb.replace_edges_by_phoval_fits(plot=False)


### Random walk parameters
nhops = 100 #nb of hops
incr = 0.002 #increment for each hop
startpoint = 'upper left' #starting point (values can be 'medium', 'upper right', 'lower right', 'lower left', 'upper left', 
                    # or some coordinates d+ and d- separated by a ';' - e.g. '9.771;9.709')
tolerance = 5e-9 #tolerance for the acceptance of a phoval in the LB (to avoid numercial artifacts)



lb.explore_one_step(incr, startpoint, nhops, tolerance)
lb.plot_last_step()



#### TBD: change self.NN as Ncheck in exploration


#### TESTS (Deprecated)

# print(lb.is_in_band((0.8-5.2j)))
# print(lb.dist_to_band((0.8-5.2j)))
# print(lb.is_in_band((0)))
# print(lb.dist_to_band(0))
# print(lb.is_in_band((-4-4j)))
# print(lb.dist_to_band(-4-4j))
# print(lb.dist_phoval_to_band(lb.phoval_inner.x, 100))
# print(lb.dist_phoval_to_band(lb.phoval_outer.x, 100))
# print(lb.dist_phoval_to_band(0.5*(np.array(lb.phoval_outer.x)+np.array(lb.phoval_inner.x)), 100))

# lb.explore_one_step(1, 0, 0) 
