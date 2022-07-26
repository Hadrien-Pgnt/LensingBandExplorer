# Lensing Band Explorer

See section 2.5 of [1] for a definition of the Kerr lensing bands (and App. A of the same paper for analytical developments about how their computation).
Numerical computation of the lensing bands is done here using modified code from AART [2].
 
 # 1. Direct problem: allowed phovals in a Lensing Band (LB)
 
 The purpose of this part of the code is the following: we suppose that we are given a LB - characterized by a value of spin, a value of inclination, and an order (typically we are interested in the n=2 image); and we attempt to answer the question: What region of the (d+, d-) plane (maximal/minimal diameter) is described by the phovals which lie within the lensing band ? See section 5.4 of [1] for examples and further details.
(and App. B of the same paper for the description of the random walk algorithm used to compute this region efficiently)

`allowedphoval_randomwalk.py` contains the main class (LensingBand) used for the computation, and all useful functions acting on that class.

`lensingband_aart.py` is the code modified from AART [2], used to compute the LB from spin, inclination and order at the very start.

`allowedphoval_util.py` gives an example of utilization of the code. typically, it should be run several times (with various starting points/increments/...), each run (called a "step") adding further data ; until sufficient coverage of the region is achieved.

The Edge_data and Steps_data directories are useful to save/load data about the LB edges and the data from previously computed steps, respectively. 

# 2. Inverse problem: lensing bands compatible with a diameter measurement

The purpose of this part of the code is the following: we suppose that we are given a (d+, d-) measurement of a photon ring (typically the n=2); and we attempt to answer the question: What values of spin and inclination yield lensing bands which contain a phoval with these extremal diameters ? 
(we use another random walk algorithm used to compute this region efficiently)

Furthermore, astrophysical profiles yield photon ring diameters which lie in a subregion of the "allowed phoval region" for one fixed value of spin & inclination - see section 5.4 of [1]. This subregion can be approximated by a box in the (d+, d-) plane stretching from the inner to the outer edge, described with 3 parameters smin, smax, deltamax. 
We also ask: with fixed box parameters, what values of spin and inclination yield boxes containing our "measured" values ?
These values are considered to be the "astrophysically realistic" ones.

`crit_curve.py` contains code that computes min/max diameter values for Kerr critical curves (for a single spin & inclination value, or on a grid), and perform phoval fits for these critical curves (with or without constraints to keep the min/max diameter fixed).

`crit_curve_map_(a,i).npy` and `crit_curve_map_(d+,d-).npy` contains precomputed data on a (spin, incl) -> (d+,d-) map using critical curve diameter values. Current precision can be improved by recomputing these files using `crit_curve.py` (a commented piece of code does this in `compatibleLBs_util.py`)

`lensingband_aart.py` is the code modified from AART [2], used to compute a lensing band (described by its edges) from spin, inclination and order.

`lensingband.py` contains a class "LensingBand" which is used for computation on the LBs (e.g. phoval fit of the edges, or test if a phoval is in the LB)

`compatibleLBs_randomwalk.py` contains the main class (DiameterMeasurement) used for the computation, and all useful functions acting on that class.

`compatibleLBs_util.py` gives an example of utilization of the code. It includes an initialization part, a random walk part for our first question, and a brute force part for the astrophysical subregion. Typically, the random walk part should be run several times (with various starting points/increments/...), each run (called a "step") adding further data ; until sufficient coverage of the region is achieved.

The Edge_data and Steps_data directories are useful to save/load data about the LB edges and the data from previously computed steps, respectively. 


[1] Paugnat, H., Lupsasca, A., Vincent, F., Wielgus, M. (2022)
Photon ring test of the Kerr hypothesis: variation in the ring shape
https://arxiv.org/abs/2206.02781

[2] Cardenas-Avendano, A., Zhu, H. & Lupsasca, A. 
Adaptive Analytical Ray-Tracing of Black Hole Photon Rings. 
To be released on arXiv.
