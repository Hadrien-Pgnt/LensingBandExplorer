# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:06:50 2022

@author: hadri
"""

from compatibleLBs_randomwalk import DiameterMeasurement

dm = DiameterMeasurement(10.318483143214817, 10.235834607023905, 2, 1000, 0.45, 50)
# these values of d+, d- correspond to the critical curve for a=0.5, i=45°

# dm = DiameterMeasurement(9.833778588299497, 9.7317131865179, 2, 1000, 0.9, 20)
# # these values of d+, d- correspond to the critical curve for a=0.94, i=17°

dm.guess_spin_incl_from_crit_curve()
print(dm.spin_guess, dm.incl_guess)