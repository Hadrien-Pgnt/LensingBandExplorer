"""
Authors: Alejandro Cardenas-Avendano, Alex Lupsasca & Hengrui Zhou  
e-mail: cardenas-avendano [at] princeton [dot] edu

This code computes the so called lensing bands 
(e.g., Fig 3 in 2008.03879). All these expressions 
appear in (P1) 1910.12873 & (P2) 1910.12881

V1. 07/10/2021
"""

import numpy as np
from numpy.lib.scimath import sqrt
from scipy.integrate import cumtrapz
from scipy.spatial import ConvexHull
from scipy import optimize
from scipy.special import ellipk
from scipy.special import ellipkinc as ellipf
from matplotlib import path

def cbrt(x):
    '''
    Cubic root
    :param x: number to compute cbrt
    '''
    if x.imag==0:
        return np.cbrt(x)
    else:
        return x**(1/3)

def nlayers(s,a,thetao,thetad,alpha,betai,mbar):
    '''
    Computes the nt-layer
    :param s: Type of ray. s>1 (Rays arrive outside) and s<1 (Rays arrive inside)
              the critical curve. We have to solve for this parameter. 
    :param a: BH's spin (-1<0<1)
    :param thetao: Observers angle [degrees]
    :param thetad: Angle of the disk [degrees]
    :param alpha: Bardeen's coordinate alpha
    :param betai: Bardeen's coordinate beta
    :param mbar: Label of the observed rings 

    :return: 
    '''
    thetao=thetao*np.pi/180
    thetad=thetad*np.pi/180
    alpha= s*alpha
    beta= s*betai

    #Angular Turning points encountered along the trajectory
    m= mbar + np.heaviside(betai,0) # Eq. (82 P1)

    # Photon conserved quantities
    # Eqs. (55 P1)
    lamb = -alpha*np.sin(thetao) 
    eta = (alpha**2 - a**2)*np.cos(thetao)**2+beta**2
    
    nutheta=np.sign(betai)
    
    # Radial Roots and Integrals 
    AAA = a**2 - eta - lamb**2 # Eq. (79 P2)
    BBB = 2*(eta + (lamb - a)**2) # Eq. (80 P2)
    CCC = -a**2 *eta # Eq. (81 P2)
    
    P = -(AAA**2/12) - CCC # Eq. (85 P2)
    Q = -(AAA/3) *((AAA/6)**2 - CCC) - BBB**2/8  # Eq. (86 P2)

    Delta3 = -4 *P**3 - 27*Q**2 # Eq. (92 P2)
    
    xi0 = np.real(cbrt(-(Q/2) + sqrt(-(Delta3/108))) + cbrt(-(Q/2) - sqrt(-(Delta3/108))) - AAA/3) # Eq. (87 P2)
    z = sqrt(xi0/2) # Eq. (94 P2)
   
    r1 = -z - sqrt(-(AAA/2) - z**2 + BBB/(4*z)) # Eq. (95a P2)
    r2 = -z + sqrt(-(AAA/2) - z**2 + BBB/(4*z)) # Eq. (95b P2)
    r3 = z - sqrt(-(AAA/2) - z**2 - BBB/(4*z))  # Eq. (95c P2)
    r4 = z + sqrt(-(AAA/2) - z**2 - BBB/(4*z))  # Eq. (95d P2)
    
    DeltaTheta = 1/2 *(1 - (eta + lamb**2)/a**2) # Eq. (19 P2)
    
    # Roots of angular potentail
    # Eqs. (B9 P2)
    uP = DeltaTheta + sqrt(DeltaTheta**2 + eta/a**2) # Eq. (19 P2)
    uM = DeltaTheta - sqrt(DeltaTheta**2 + eta/a**2) # Eq. (19 P2)
    
    # Eqs. (B9 P2)
    r21 = r2 - r1 
    r31 = r3 - r1
    r32 = r3 - r2
    r41 = r4 - r1
    r42 = r4 - r2
    r43 = r4 - r3
    
    # Outer and inner horizons
    # Eqs. (2 P2)
    rP = 1 + np.sqrt(1 - a**2)
    rM = 1 - np.sqrt(1 - a**2)
    
    # Eqs. (B10 P2)
    a1=sqrt(-(r43**2/4))
    # Eqs. (B10 P2)
    b1=(r3 + r4)/2
    
    #Elliptic Parameter
    # Eqs. (B13 P2)
    k = (r32*r41)/(r31*r42)

    AA = np.real(sqrt(a1**2 + (b1 - r2)**2)) # Eqs. (B56 P2)
    BB = np.real(sqrt(a1**2 + (b1 - r1)**2)) # Eqs. (B56 P2)

    # This parameter is real and less the unity
    k3 = np.real(((AA + BB)**2 - r21**2)/(4*AA*BB)) # Eqs. (B59 P2)
    
    # Eqs. (20 P1)
    Gtheta = 1/(sqrt(-uM)*a)*(2*m*ellipk(uP/uM) -nutheta*ellipf(np.arcsin(np.cos(thetao)/np.sqrt(uP)), uP/uM) + nutheta*(-1)**m*ellipf(np.arcsin(np.cos(thetad)/np.sqrt(uP)), uP/uM))
    
    if s>1:
        # Eqs. (A10 P1)
        Q1= 4/sqrt(r31*r42)*ellipf(np.arcsin(sqrt(r31/r41)), k)
        return Q1-Gtheta
    else:
        if k3<1:
            # Eqs. (A11 P1)
            Q2=1/sqrt(AA*BB)*(ellipf(np.arccos((AA - BB)/(AA + BB)), k3) - ellipf(np.arccos((AA *(rP - r1) - BB*(rP - r2))/(AA*(rP - r1) + BB*(rP - r2))), k3))
        else:
            Q2=np.nan
    
        return Q2-Gtheta

def spacedmarks(x, y, Nmarks):

    """
    Computes the arch-length
    :param x: x point
    :param y: y point
    :param Nmarks: Number of marks 

    :returns: position of the x and y markers
    """
    dydx = np.gradient(y, x[0],edge_order=2)
    dxdx = np.gradient(x, x[0],edge_order=2)
    arclength = cumtrapz(sqrt(dydx**2 + dxdx**2), initial=0)
    marks = np.linspace(0, max(arclength), Nmarks)
    markx = np.interp(marks, arclength, x)
    marky = np.interp(markx, x, y)
    return markx, marky
    

def Shadow(a,angle):

    thetao = angle * np.pi/180
        
    rM = 2*(1 + np.cos(2/3 *np.arccos(-(a))))
    rP = 2*(1 + np.cos(2/3 *np.arccos(a)))
    
    #print(rM,rP)
    
    r=np.linspace(rM,rP,int(1e7))
    
    lam = a + r/a *(r - (2 *(r**2 - 2*r + a**2))/(r - 1))
    eta = r**3/a**2 *((4*(r**2 - 2*r + a**2))/(r - 1)**2 - r)

    alpha=-lam/np.sin(thetao)
    beta=eta + a**2 *np.cos(thetao)**2 - lam**2*np.tan(thetao)**(-2)
    
    mask=np.where(beta>0)
    r=r[mask]
    
    rmin=min(r)+1e-12
    rmax=max(r)-+1e-12
    
    r=np.linspace(rmin,rmax,int(1e6))
        
    lam = a + r/a *(r - (2 *(r**2 - 2*r + a**2))/(r - 1))
    eta = r**3/a**2 *((4*(r**2 - 2*r + a**2))/(r - 1)**2 - r)

    alpha=-lam/np.sin(thetao)
    beta=eta + a**2 *np.cos(thetao)**2 - lam**2*np.tan(thetao)**(-2)

    return alpha, sqrt(beta)


def grid_nlayer(lims,hull,hull2,dx):
    
    x = np.linspace(-lims, lims, int(lims/dx))
    y = np.linspace(-lims, lims, int(lims/dx))
    
    mesh = np.array(np.meshgrid(x , y))
    grid=mesh.T.reshape(-1, 2)
    
    p1 = path.Path(hull.points)
    p2 = path.Path(hull2.points)
    
    mask1=p1.contains_points(grid)
    mask2=np.invert(p2.contains_points(grid))
    
    indexes=mask1*mask2

    return grid[indexes,:]
