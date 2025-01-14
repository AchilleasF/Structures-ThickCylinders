### Linear elastic bifurcation buckling load of shear-flexible cylinder under normal pressure
### A BC2f - BC2f set of boundary conditions is considered at the edges of the cylinder
# Last modified by Achilleas Filippidis on 18/10/2024

# Necessary packages to run this script
from math import *
import numpy as np

def pthick_mn(E,nu,k,r,t,L,m,n):
    """
    Evaluates thick non-shallow shell theory
    bifurcation buckling load for a cylinder under external pressure.

    :param float E: Young's modulus of the material
    :param float nu: Poisson's ratio of the material
    :param float k: Shear correction factor of the material. A value of 5/6 is recommended.
    :param float r: Cylinder's radius
    :param float t: Cylinder's wall thickness
    :param float t: Cylinder's length
    :param float m: No of meridional half-waves. While float values are mathematically admissible, only integers are physically meaningful values
    :param float n: No of circumferential full-waves. While float values are mathematically admissible, only integers are physically meaningful values
    :return: A linearised prediction of the external pressure bifurcation buckling load for a shear-flexible cylinder
    """

    # Evaluate numerator terms for elastic bifurcation buckling load fraction
    pmn_n = np.zeros([6,6])
    pmn_n[5,0] = 2*t**3*k*r**9
    pmn_n[4,1] = 10*t**3*r**7*k
    pmn_n[4,0] = 2*t**2*r**7*(12*k**2*r**2 + k**2*t**2 + 2*nu + 2)
    pmn_n[3,2] = 20*t**3*r**5*k
    pmn_n[3,1] = -t*r**5*(k**2*nu*t**3 - 96*k**2*r**2*t - 5*k**2*t**3 + 4*k*nu*t**2 + 8*k*t**2 - 8*nu*t - 8*t)
    pmn_n[3,0] = -t*r**5*(24*k*nu**2*r**2 - 48*k*nu*r**2 - 4*k*nu*t**2 - 72*k*r**2 - 4*k*t**2)
    pmn_n[2,3] = 20*r**3*k*t**3
    pmn_n[2,2] = -2*r**3*(k**2*nu*t**4 - 72*k**2*r**2*t**2 - 2*k**2*t**4 + 4*k*nu*t**3 + 10*k*t**3 - 2*nu*t**2 - 2*t**2)
    pmn_n[2,1] = -2*r**3*(24*k**2*nu*r**2*t**2 + 48*k**2*r**2*t**2 + 12*k*nu**2*r**2*t - 24*k*nu*r**2*t - 4*k*nu*t**3 - 36*k*r**2*t - 5*k*t**3)
    pmn_n[2,0] = -2*r**3*(144*k**2*nu**2*r**4 + 12*k**2*nu**2*r**2*t**2 - 144*k**2*r**4 - 12*k**2*r**2*t**2)
    pmn_n[1,4] = 10*t**3*k*r
    pmn_n[1,3] = -k*t**2*r*(k*nu*t**2 - 96*k*r**2 - k*t**2 + 4*nu*t + 16*t) 
    pmn_n[1,2] = -k*t**2*r*(48*k*nu*r**2 + 144*k*r**2 - 4*nu*t - 8*t)
    pmn_n[1,1] = -k*t**2*r*(-36*k*nu*r**2 - 60*k*r**2)
    pmn_n[0,5] = 2*t**3*k/r
    pmn_n[0,4] = 2*(12*k**2*r**2*t**2 - 2*k*t**3)/r
    pmn_n[0,3] = 2*(-24*k**2*r**2*t**2 + k*t**3)/r
    pmn_n[0,2] = 24*k**2*r*t**2

    # Evaluate denominator terms for elastic bifurcation buckling load fraction
    pmn_d = np.zeros([6,6])
    pmn_d[4,1] = -4*(1 + nu)*r**8*t**2
    pmn_d[4,0] = -4*(-1 - nu)*r**8*t**2
    pmn_d[3,2] = -16*(1 + nu)*r**6*t**2
    pmn_d[3,1] = 16*(1 + nu)*r**6*t**2 + 4*r**6*t*(1 + nu)*(6*k*nu*r**2 - 18*k*r**2 - k*t**2)
    pmn_d[3,0] = -4*r**6*t*(1 + nu)*(6*k*nu*r**2 - 18*k*r**2 - k*t**2)
    pmn_d[2,3] = -24*(1 + nu)*r**4*t**2
    pmn_d[2,2] = 24*(1 + nu)*r**4*t**2 + 2*r**4*(1 + nu)*(36*k*nu*r**2*t + k*nu*t**3 - 108*k*r**2*t - 5*k*t**3)
    pmn_d[2,1] = -2*r**4*(1 + nu)*(36*k*nu*r**2*t + k*nu*t**3 - 108*k*r**2*t - 5*k*t**3) + 2*r**4*(1 + nu)*(144*k**2*nu*r**4 + 12*k**2*nu*r**2*t**2 - 144*k**2*r**4 - 12*k**2*r**2*t**2)
    pmn_d[2,0] = -2*r**4*(1 + nu)*(144*k**2*nu*r**4 + 12*k**2*nu*r**2*t**2 - 144*k**2*r**4 - 12*k**2*r**2*t**2)
    pmn_d[1,4] = -16*(1 + nu)*r**2*t**2
    pmn_d[1,3] = 16*(1 + nu)*r**2*t**2 - 4*r**2*(1 + nu)*(-18*k*nu*r**2*t - k*nu*t**3 + 54*k*r**2*t + 2*k*t**3)
    pmn_d[1,2] = 4*r**2*(1 + nu)*(-18*k*nu*r**2*t - k*nu*t**3 + 54*k*r**2*t + 2*k*t**3) - 4*r**2*(1 + nu)*(3*k**2*nu**2*r**2*t**2 - 144*k**2*nu*r**4 - 6*k**2*nu*r**2*t**2 + 144*k**2*r**4 + 15*k**2*r**2*t**2)
    pmn_d[1,1] = 4*r**2*(1 + nu)*(3*k**2*nu**2*r**2*t**2 - 144*k**2*nu*r**4 - 6*k**2*nu*r**2*t**2 + 144*k**2*r**4 + 15*k**2*r**2*t**2)
    pmn_d[0,5] = -4*(1 + nu)*t**2
    pmn_d[0,4] = -4*((-1 - nu)*t + 12*(1 + nu)*k*r**2)*t + 2*(1 + nu)*t*(12*k*nu*r**2 + k*nu*t**2 - 12*k*r**2 - k*t**2)
    pmn_d[0,3] = -48*(-1 - nu)*k*r**2*t + 2*((-1 - nu)*t + 12*(1 + nu)*k*r**2)*(12*k*nu*r**2 + k*nu*t**2 - 12*k*r**2 - k*t**2)
    pmn_d[0,2] = 24*(-1 - nu)*k*r**2*(12*k*nu*r**2 + k*nu*t**2 - 12*k*r**2 - k*t**2)

    # mbar definition
    mbar = m*pi/L
    # Assemble fraction for pmn
    pmn_n_sum, pmn_d_sum = 0, 0
    for i in range(0,6):
        for j in range(0,6):
            pmn_n_sum += pmn_n[i,j]*(mbar**(2*i))*(n**(2*j))
            pmn_d_sum += pmn_d[i,j]*(mbar**(2*i))*(n**(2*j))

    # Evaluate and return bifurcation buckling pressure
    pmn = E*t*(pmn_n_sum/pmn_d_sum)/r
    return pmn