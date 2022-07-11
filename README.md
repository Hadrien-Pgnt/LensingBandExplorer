# Lensing Band Explorer

See section 2.5 of [1] for a definition of the Kerr lensing bands (and App. A of the same paper for analytical developments about how their computation).
Numerical computation of the lensing bands is done here using modified code from AART [2].
 
 # 1. Direct problem: allowed phovals in a Lensing Band (LB)
 
 The purpose of this part of the code is the following: we suppose that we are given a LB - characterized by a value of spin, a value of inclination, and an order (typically we are interested in the n=2 image); and we attempt to answer the question: What region of the (d+, d-) plane (maximal/minimal diameter) is described by the phovals which lie within the lensing band ? See section 5.4 of [1] for examples and further details.
(and App. B of the same paper for the description of the random walk algorithm used to compute this region efficiently)

`allowedphoval_randomwalk.py` contains the main class (LensingBand) used for the computation, and most useful functions (acting on that class)

`lensingband_aart.py` is the code modified from AART [2], used to compute the LB from spin, inclination and order at the very start.

`allowedphoval_util.py` gives an example of utilization of the code. typically, it should be run several times (with various starting points/increments/...), each run (called a "step") adding further data ; until sufficient coverage of the region is achieved.

The Edge_data and Steps_data directories are useful to save/load data about the LB edges and the data from previously computed steps, respectively. 


[1] Paugnat, H., Lupsasca, A., Vincent, F., Wielgus, M. (2022)
Photon ring test of the Kerr hypothesis: variation in the ring shape
https://arxiv.org/abs/2206.02781

[2] Cardenas-Avendano, A., Zhu, H. & Lupsasca, A. 
Adaptive Analytical Ray-Tracing of Black Hole Photon Rings. 
To be released on arXiv.
