### Linear elastic stress analysis of shear-flexible cylinder under edge load
### A BC1r boundary condition is considered for its base and a BC2f for the top edge
# Last modified by Achilleas Filippidis on 18/10/2024

# Necessary packages to run this script
from math import *
import numpy as np
import cmath as cm
import pandas as pd
import matplotlib.pyplot as plt

# Modifying the following parameters will lead to a different model
# Thick and thin solutions will still be comparable, but the ABAQUS FE results only refer to the original model parameters
# Material parameters
E = 200000.0
nu = 0.3
k = 5/6
# Geometric parameters
r = 5
t = 1
omega = 5
L = omega*sqrt(r*t)
Z = L**2/(r*t)*sqrt(1-nu*nu)
# Loading parameters
pn = 0
Nz0 = -1
# Constants
C = E*t/(1-nu**2)
D = E*t**3/12/(1-nu*nu)
G = E*t/2/(1+nu)
# Plot controls
LW = 2
MS = 4
FS = 9
# Colors
black = '#000000'
deepblue = '#0A1172'
blue = '#1338BE'
lightblue = '#0492C2'
deepred = '#8D0000'
red = '#B90E0A'
lightred = '#FF8282'
deepgray = '#373737'
gray = '#594D5B'
lightgray = '#808080'
# Meridional discretisation for plotting
npoints = 200
zs = np.linspace(0,L,num=200)
invzs = np.linspace(L,0,num=200)
# Load abaqus data
abaqus = pd.read_csv('abaqus_ladata.csv')
abaqus.sigmaphi_i = abaqus.N_S11_top; abaqus.sigmaphi_m = (abaqus.N_S11_bot+abaqus.N_S11_top)/2; abaqus.sigmaphi_o = abaqus.N_S11_bot
abaqus.sigmatheta_i = abaqus.N_S22_top; abaqus.sigmatheta_m = (abaqus.N_S22_bot+abaqus.N_S22_top)/2; abaqus.sigmatheta_o = abaqus.N_S22_bot



###################################
### Thick shell theory solution ###
###################################
# Manual computation of eigenvalues and eigenvectors
a = (1/r/sqrt(2))*sqrt(sqrt(12*(1-nu*nu)*(r/t)**2) + (1+nu)/k)
b = (1/r/sqrt(2))*sqrt(sqrt(12*(1-nu*nu)*(r/t)**2) - (1+nu)/k)
# Eigenvalues
Ds = np.array([  a + b*1j,
                 a - b*1j,
               - a + b*1j,
               - a - b*1j])
# Eigenvectors
Vs = np.array([[1, 1, 1, 1],
               [Ds[0], Ds[1], Ds[2], Ds[3]],
               [Ds[0]/(1-D*Ds[0]*Ds[0]/k/G/t), Ds[1]/(1-D*Ds[1]*Ds[1]/k/G/t), Ds[2]/(1-D*Ds[2]*Ds[2]/k/G/t), Ds[3]/(1-D*Ds[3]*Ds[3]/k/G/t)],
               [Ds[0]*Ds[0]/(1-D*Ds[0]*Ds[0]/k/G/t), Ds[1]*Ds[1]/(1-D*Ds[1]*Ds[1]/k/G/t), Ds[2]*Ds[2]/(1-D*Ds[2]*Ds[2]/k/G/t), Ds[3]*Ds[3]/(1-D*Ds[3]*Ds[3]/k/G/t)]])

# Coefficients of unknown constants at z = 0
# For normal displacement w
w_c1_0 = Vs[0,0]*cm.exp(Ds[0]*0)
w_c2_0 = Vs[0,1]*cm.exp(Ds[1]*0)
w_c3_0 = Vs[0,2]*cm.exp(Ds[2]*0)
w_c4_0 = Vs[0,3]*cm.exp(Ds[3]*0)
# For first derivative of normal displacement w
wp_c1_0 = Vs[1,0]*cm.exp(Ds[0]*0)
wp_c2_0 = Vs[1,1]*cm.exp(Ds[1]*0)
wp_c3_0 = Vs[1,2]*cm.exp(Ds[2]*0)
wp_c4_0 = Vs[1,3]*cm.exp(Ds[3]*0)
# For meridional rotation beta
b_c1_0 = Vs[2,0]*cm.exp(Ds[0]*0)
b_c2_0 = Vs[2,1]*cm.exp(Ds[1]*0)
b_c3_0 = Vs[2,2]*cm.exp(Ds[2]*0)
b_c4_0 = Vs[2,3]*cm.exp(Ds[3]*0)
# For first derivative of meridional rotation beta
bp_c1_0 = Vs[3,0]*cm.exp(Ds[0]*0)
bp_c2_0 = Vs[3,1]*cm.exp(Ds[1]*0)
bp_c3_0 = Vs[3,2]*cm.exp(Ds[2]*0)
bp_c4_0 = Vs[3,3]*cm.exp(Ds[3]*0)
# For second derivative of meridional rotation beta
bpp_c1_0 = Ds[0]*Vs[3,0]*cm.exp(Ds[0]*0)
bpp_c2_0 = Ds[1]*Vs[3,1]*cm.exp(Ds[1]*0)
bpp_c3_0 = Ds[2]*Vs[3,2]*cm.exp(Ds[2]*0)
bpp_c4_0 = Ds[3]*Vs[3,3]*cm.exp(Ds[3]*0)

# Coefficients of unknown constants at z = L
# For normal displacement w
w_c1_L = Vs[0,0]*cm.exp(Ds[0]*L)
w_c2_L = Vs[0,1]*cm.exp(Ds[1]*L)
w_c3_L = Vs[0,2]*cm.exp(Ds[2]*L)
w_c4_L = Vs[0,3]*cm.exp(Ds[3]*L)
# For first derivative of normal displacement w
wp_c1_L = Vs[1,0]*cm.exp(Ds[0]*L)
wp_c2_L = Vs[1,1]*cm.exp(Ds[1]*L)
wp_c3_L = Vs[1,2]*cm.exp(Ds[2]*L)
wp_c4_L = Vs[1,3]*cm.exp(Ds[3]*L)
# For meridional rotation beta
b_c1_L = Vs[2,0]*cm.exp(Ds[0]*L)
b_c2_L = Vs[2,1]*cm.exp(Ds[1]*L)
b_c3_L = Vs[2,2]*cm.exp(Ds[2]*L)
b_c4_L = Vs[2,3]*cm.exp(Ds[3]*L)
# For first derivative of meridional rotation beta
bp_c1_L = Vs[3,0]*cm.exp(Ds[0]*L)
bp_c2_L = Vs[3,1]*cm.exp(Ds[1]*L)
bp_c3_L = Vs[3,2]*cm.exp(Ds[2]*L)
bp_c4_L = Vs[3,3]*cm.exp(Ds[3]*L)
# For second derivative of meridional rotation beta
bpp_c1_L = Ds[0]*Vs[3,0]*cm.exp(Ds[0]*L)
bpp_c2_L = Ds[1]*Vs[3,1]*cm.exp(Ds[1]*L)
bpp_c3_L = Ds[2]*Vs[3,2]*cm.exp(Ds[2]*L)
bpp_c4_L = Ds[3]*Vs[3,3]*cm.exp(Ds[3]*L)

# Particular solution for normal displacement w at (constant for any z)
wm = pn*r*r/t/E - Nz0*nu*r/t/E
# Particular solution for meridional rotation beta (constant for any z)
bm = 0.0

# LHS coefficients matrix for unknown constants
Am = np.array([[w_c1_0, w_c2_0, w_c3_0, w_c4_0],
               [bp_c1_0, bp_c2_0, bp_c3_0, bp_c4_0],
               [w_c1_L, w_c2_L, w_c3_L, w_c4_L],
               [b_c1_L, b_c2_L, b_c3_L, b_c4_L]])
# RHS vector, corresponding to the boundary conditions and membrane solution
Bv = np.array([-wm, -bm, -wm, -bm])
## Solve for unknown coefficients
Cis = np.linalg.solve(Am, Bv)
c1 = Cis[0]; c2 = Cis[1]; c3 = Cis[2]; c4 = Cis[3]

# Bending theory solution for normal displacement w
uL = Nz0/C*L-nu/r*(c1*w_c1_L/Ds[0] + c2*w_c2_L/Ds[1] + c3*w_c3_L/Ds[2] + c4*w_c4_L/Ds[3] + wm*L)
u, up, w, wp, beta, betap, betapp = np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints)
Nphi, Ntheta, Mphi, Mtheta = np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints)
sigmaphi_i, sigmaphi_m, sigmaphi_o, sigmatheta_i, sigmatheta_m, sigmatheta_o = np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints)
for i in range(0,npoints):
    z = zs[i]
    w[i] = c1*Vs[0,0]*cm.exp(Ds[0]*z) + c2*Vs[0,1]*cm.exp(Ds[1]*z) + c3*Vs[0,2]*cm.exp(Ds[2]*z) + c4*Vs[0,3]*cm.exp(Ds[3]*z) + wm
    wp[i] = c1*Vs[1,0]*cm.exp(Ds[0]*z) + c2*Vs[1,1]*cm.exp(Ds[1]*z) + c3*Vs[1,2]*cm.exp(Ds[2]*z) + c4*Vs[1,3]*cm.exp(Ds[3]*z)
    beta[i] = c1*Vs[2,0]*cm.exp(Ds[0]*z) + c2*Vs[2,1]*cm.exp(Ds[1]*z) + c3*Vs[2,2]*cm.exp(Ds[2]*z) + c4*Vs[2,3]*cm.exp(Ds[3]*z) + bm
    betap[i] = c1*Vs[3,0]*cm.exp(Ds[0]*z) + c2*Vs[3,1]*cm.exp(Ds[1]*z) + c3*Vs[3,2]*cm.exp(Ds[2]*z) + c4*Vs[3,3]*cm.exp(Ds[3]*z)
    betapp[i] = c1*Ds[0]*Vs[3,0]*cm.exp(Ds[0]*z) + c2*Ds[1]*Vs[3,1]*cm.exp(Ds[1]*z) + c3*Ds[2]*Vs[3,2]*cm.exp(Ds[2]*z) + c4*Ds[3]*Vs[3,3]*cm.exp(Ds[3]*z)
    u[i] = Nz0/C*z-nu/r*(c1*Vs[0,0]*cm.exp(Ds[0]*z)/Ds[0] + c2*Vs[0,1]*cm.exp(Ds[1]*z)/Ds[1] + c3*Vs[0,2]*cm.exp(Ds[2]*z)/Ds[2] + c4*Vs[0,3]*cm.exp(Ds[3]*z)/Ds[3] + wm*z) - uL
    up[i] = Nz0/C-nu/r*w[i]
    Nphi[i] = C*(up[i]+nu/r*w[i])
    Ntheta[i] = C*(w[i]/r+nu*up[i])
    Mphi[i] = D*betap[i]
    Mtheta[i] = nu*D*betap[i]
    sigmaphi_i[i] = Nphi[i]/t + 6*Mphi[i]/t/t
    sigmaphi_m[i] = Nphi[i]/t
    sigmaphi_o[i] = Nphi[i]/t - 6*Mphi[i]/t/t
    sigmatheta_i[i] = Ntheta[i]/t + 6*Mtheta[i]/t/t
    sigmatheta_m[i] = Ntheta[i]/t
    sigmatheta_o[i] = Ntheta[i]/t - 6*Mtheta[i]/t/t


##################################
### Thin shell theory solution ###
##################################
# The following thin shell theory bending solution is based on the following article
# Rotter, J. M., & Sadowski, A. J. (2012). Cylindrical shell bending theory for orthotropic shells under general axisymmetric pressure distributions. Engineering Structures, 42, 258-265.
# Membrane theory solution
NZm = Nz0 # Meridional (axial) membrane stress resultant
NThm = pn*r # Circumferential membrane stress resultant
wm = (r/(E*t))*(NThm - nu*NZm) # Radial midsurface displacement for membrane stress state only
wm_p = 0; # First derivative of wm w.r.t. z
wm_pp = 0; # Second derivative of wm w.r.t. z
# Linear bending half-wavelength
lamda = pi*sqrt(r*t)*(3.0*(1.0 - nu*nu))**(-0.25)

# Coefficients (their derivatives and anti-derivatives) of the radial
# displacement field for z = 0
c1_0 = exp(-pi*0/lamda)*cos(pi*0/lamda)
c2_0 = exp(-pi*0/lamda)*sin(pi*0/lamda)
c3_0 = exp(pi*0/lamda)*cos(pi*0/lamda)
c4_0 = exp(pi*0/lamda)*sin(pi*0/lamda)
c1p_0 = -(pi/lamda)*exp(-pi*0/lamda)*(cos(pi*0/lamda)+sin(pi*0/lamda))
c2p_0 = (pi/lamda)*exp(-pi*0/lamda)*(cos(pi*0/lamda)-sin(pi*0/lamda))
c3p_0 = (pi/lamda)*exp(pi*0/lamda)*(cos(pi*0/lamda)-sin(pi*0/lamda))
c4p_0 = (pi/lamda)*exp(pi*0/lamda)*(cos(pi*0/lamda)+sin(pi*0/lamda))
c1pp_0 = 2.0*(pi/lamda)**2*exp(-pi*0/lamda)*sin(pi*0/lamda)
c2pp_0 = -2.0*(pi/lamda)**2*exp(-pi*0/lamda)*cos(pi*0/lamda)
c3pp_0 = -2.0*(pi/lamda)**2*exp(pi*0/lamda)*sin(pi*0/lamda)
c4pp_0 = 2.0*(pi/lamda)**2*exp(pi*0/lamda)*cos(pi*0/lamda)
c1ppp_0 = 2.0*(pi/lamda)**3*exp(-pi*0/lamda)*(cos(pi*0/lamda)-sin(pi*0/lamda))
c2ppp_0 = 2.0*(pi/lamda)**3*exp(-pi*0/lamda)*(cos(pi*0/lamda)+sin(pi*0/lamda))
c3ppp_0 = -2.0*(pi/lamda)**3*exp(pi*0/lamda)*(cos(pi*0/lamda)+sin(pi*0/lamda))
c4ppp_0 = 2.0*(pi/lamda)**3*exp(pi*0/lamda)*(cos(pi*0/lamda)-sin(pi*0/lamda))
c1i_0 = 0.5*lamda/pi*(1-exp(-pi*0/lamda)*(cos(pi*0/lamda)-sin(pi*0/lamda)))
c2i_0 = 0.5*lamda/pi*(1-exp(-pi*0/lamda)*(cos(pi*0/lamda)+sin(pi*0/lamda)))
c3i_0 = 0.5*lamda/pi*(-1+exp(pi*0/lamda)*(sin(pi*0/lamda)+cos(pi*0/lamda)))
c4i_0 = 0.5*lamda/pi*(1+exp(pi*0/lamda)*(sin(pi*0/lamda)-cos(pi*0/lamda)))

# Coefficients (their derivatives and anti-derivatives) of the radial
# displacement field for z = L
c1_L = exp(-pi*L/lamda)*cos(pi*L/lamda)
c2_L = exp(-pi*L/lamda)*sin(pi*L/lamda)
c3_L = exp(pi*L/lamda)*cos(pi*L/lamda)
c4_L = exp(pi*L/lamda)*sin(pi*L/lamda)
c1p_L = -(pi/lamda)*exp(-pi*L/lamda)*(cos(pi*L/lamda)+sin(pi*L/lamda))
c2p_L = (pi/lamda)*exp(-pi*L/lamda)*(cos(pi*L/lamda)-sin(pi*L/lamda))
c3p_L = (pi/lamda)*exp(pi*L/lamda)*(cos(pi*L/lamda)-sin(pi*L/lamda))
c4p_L = (pi/lamda)*exp(pi*L/lamda)*(cos(pi*L/lamda)+sin(pi*L/lamda))
c1pp_L = 2.0*(pi/lamda)**2*exp(-pi*L/lamda)*sin(pi*L/lamda)
c2pp_L = -2.0*(pi/lamda)**2*exp(-pi*L/lamda)*cos(pi*L/lamda)
c3pp_L = -2.0*(pi/lamda)**2*exp(pi*L/lamda)*sin(pi*L/lamda)
c4pp_L = 2.0*(pi/lamda)**2*exp(pi*L/lamda)*cos(pi*L/lamda)
c1ppp_L = 2.0*(pi/lamda)**3*exp(-pi*L/lamda)*(cos(pi*L/lamda)-sin(pi*L/lamda))
c2ppp_L = 2.0*(pi/lamda)**3*exp(-pi*L/lamda)*(cos(pi*L/lamda)+sin(pi*L/lamda))
c3ppp_L = -2.0*(pi/lamda)**3*exp(pi*L/lamda)*(cos(pi*L/lamda)+sin(pi*L/lamda))
c4ppp_L = 2.0*(pi/lamda)**3*exp(pi*L/lamda)*(cos(pi*L/lamda)-sin(pi*L/lamda))
c1i_L = 0.5*lamda/pi*(1-exp(-pi*L/lamda)*(cos(pi*L/lamda)-sin(pi*L/lamda)))
c2i_L = 0.5*lamda/pi*(1-exp(-pi*L/lamda)*(cos(pi*L/lamda)+sin(pi*L/lamda)))
c3i_L = 0.5*lamda/pi*(-1+exp(pi*L/lamda)*(sin(pi*L/lamda)+cos(pi*L/lamda)))
c4i_L = 0.5*lamda/pi*(1+exp(pi*L/lamda)*(sin(pi*L/lamda)-cos(pi*L/lamda)))

# Construction and solution of matrix system
a1_0 = c1_0; a2_0 = c2_0; a3_0 = c3_0; a4_0 = c4_0
a1p_0 = c1p_0; a2p_0 = c2p_0; a3p_0 = c3p_0; a4p_0 = c4p_0
a1pp_0 = c1pp_0; a2pp_0 = c2pp_0; a3pp_0 = c3pp_0; a4pp_0 = c4pp_0
a1_h = c1_L; a2_h = c2_L; a3_h = c3_L; a4_h = c4_L
a1p_h = c1p_L; a2p_h = c2p_L; a3p_h = c3p_L; a4p_h = c4p_L
a1pp_h = c1pp_L; a2pp_h = c2pp_L; a3pp_h = c3pp_L; a4pp_h = c4pp_L

# LHS coefficients matrix for unknown constants
M = np.array([[a1_0,    a2_0,    a3_0,    a4_0],
               [a1p_0,   a2p_0,   a3p_0,   a4p_0],
               [a1_h,    a2_h,    a3_h,    a4_h],
               [a1pp_h,  a2pp_h,  a3pp_h,  a4pp_h]])
# RHS vector, corresponding to the boundary conditions and membrane solution
V = np.array([-wm, -wm_p, -wm, -wm_pp])
As = np.linalg.solve(M, V); A1 = As[0]; A2 = As[1]; A3 = As[2]; A4 = As[3]

# Bending theory solution for normal displacement w
uL = Nz0/C*L-nu/r*(c1*w_c1_L/Ds[0] + c2*w_c2_L/Ds[1] + c3*w_c3_L/Ds[2] + c4*w_c4_L/Ds[3] + wm*L)
ub, ub_p, wb, wb_pp = np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints)
Nph_b, Nth_b, Mph_b, Mth_b = np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints)
sigmaphi_bi, sigmaphi_bm, sigmaphi_bo, sigmatheta_bi, sigmatheta_bm, sigmatheta_bo = np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints), np.empty(npoints)
for i in range(0,npoints):
    z = zs[i]
    wb[i] = A1*exp(-pi*z/lamda)*cos(pi*z/lamda) + A2*exp(-pi*z/lamda)*sin(pi*z/lamda) + A3*exp(pi*z/lamda)*cos(pi*z/lamda) + A4*exp(pi*z/lamda)*sin(pi*z/lamda) + wm
    wb_pp[i] = A1*2.0*(pi/lamda)**2*exp(-pi*z/lamda)*sin(pi*z/lamda) - A2*2.0*(pi/lamda)**2*exp(-pi*z/lamda)*cos(pi*z/lamda) - A3*2.0*(pi/lamda)**2*exp(pi*z/lamda)*sin(pi*z/lamda) + A4*2.0*(pi/lamda)**2*exp(pi*z/lamda)*cos(pi*z/lamda) + wm_pp
    ub[i] = z*Nz0/C - nu/r*(A1*0.5*lamda/pi*(1-exp(-pi*z/lamda)*(cos(pi*z/lamda)-sin(pi*z/lamda))) + A2*0.5*lamda/pi*(1-exp(-pi*z/lamda)*(cos(pi*z/lamda)+sin(pi*z/lamda))) + A3*0.5*lamda/pi*(-1+exp(pi*z/lamda)*(sin(pi*z/lamda)+cos(pi*z/lamda))) + A4*0.5*lamda/pi*(1+exp(pi*z/lamda)*(sin(pi*z/lamda)-cos(pi*z/lamda))) + z*wm)
    ub_p[i] = Nz0/C - nu/r*wb[i]
    Nph_b[i] = C*(ub_p[i]+nu*wb[i]/r)
    Nth_b[i] = C*(wb[i]/r+nu*up[i])
    Mph_b[i] = D*wb_pp[i]
    Mth_b[i] = nu*D*wb_pp[i]
    sigmaphi_bi[i] = Nph_b[i]/t + 6*Mph_b[i]/t/t
    sigmaphi_bm[i] = Nph_b[i]/t
    sigmaphi_bo[i] = Nph_b[i]/t - 6*Mph_b[i]/t/t
    sigmatheta_bi[i] = Nth_b[i]/t + 6*Mth_b[i]/t/t
    sigmatheta_bm[i] = Nth_b[i]/t
    sigmatheta_bo[i] = Nth_b[i]/t - 6*Mth_b[i]/t/t

########################
### Generate figures ###
########################
plt.close('all')
# Meridional edge load u displacement profile
plt.figure(1)
plt.grid()
thickline, = plt.plot(u*1e5, zs,color=lightgray,linestyle='-',linewidth=LW)
thinline, = plt.plot(-ub*1e5, invzs,color=black,linestyle='--',linewidth=0.5*LW)
abaqusline, = plt.plot(-abaqus.N_U2*1e5, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='o',markersize=MS,markerfacecolor='white',markevery=4)
plt.gca().invert_yaxis()
plt.legend([thickline,thinline,abaqusline],['Thick theory','Thin theory','ABAQUS FE'],loc='lower right', fontsize=FS, frameon=True, edgecolor=black)
plt.xlabel('$u\cdot 10^{-5}$ [mm]',fontdict={'size':FS})
plt.ylabel('Axial coordinate $z$ [mm]',fontdict={'size':FS})

# Meridional edge load displacement profile
plt.figure(2)
plt.grid()
thickline, = plt.plot(w*1e6, zs,color=lightgray,linestyle='-',linewidth=LW)
thinline, = plt.plot(wb*1e6, invzs,color=black,linestyle='--',linewidth=0.5*LW)
abaqusline, = plt.plot(abaqus.N_U1*1e6, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='o',markersize=MS,markerfacecolor='white',markevery=4)
plt.gca().invert_yaxis()
plt.legend([thickline,thinline,abaqusline],['Thick theory','Thin theory','ABAQUS FE'],loc='center left', fontsize=FS, frameon=True, edgecolor=black)
plt.xlabel('$w\cdot 10^{-6}$ [mm]',fontdict={'size':FS})
plt.ylabel('Axial coordinate $z$ [mm]',fontdict={'size':FS})

# Meridional edge load meridional stress patterns sigma_phi
plt.figure(3)
plt.grid()
thickline, = plt.plot(sigmaphi_i, zs,color=lightgray,linestyle='-',linewidth=LW)
plt.plot(sigmaphi_m, zs,color=lightgray,linestyle='-',linewidth=LW)
plt.plot(sigmaphi_o, zs,color=lightgray,linestyle='-',linewidth=LW)
thinline, = plt.plot(sigmaphi_bi, invzs,color=black,linestyle='--',linewidth=0.5*LW)
plt.plot(sigmaphi_bm, invzs,color=black,linestyle='--',linewidth=0.5*LW)
plt.plot(sigmaphi_bo, invzs,color=black,linestyle='--',linewidth=0.5*LW)
abaqusin, = plt.plot(abaqus.sigmaphi_i, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='d',markersize=MS,markerfacecolor='white',markevery=4)
abaqusmid, = plt.plot(abaqus.sigmaphi_m, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='o',markersize=MS,markerfacecolor='white',markevery=4)
abaqusout, = plt.plot(abaqus.sigmaphi_o, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='s',markersize=MS,markerfacecolor='white',markevery=4)
plt.gca().invert_yaxis()
plt.legend([thickline,thinline,abaqusin,abaqusmid,abaqusout],['Thick theory','Thin theory','ABAQUS FE (inner surf.)','ABAQUS FE (middle surf.)','ABAQUS FE (outer surf.)'],loc='center left', fontsize=FS, frameon=True, edgecolor=black)
plt.xlabel('$\sigma_\\phi$ [$N/mm^2$]',fontdict={'size':FS})
plt.ylabel('Axial coordinate $z$ [mm]',fontdict={'size':FS})

# Meridional edge load circumferential stress patterns sigma_theta
plt.figure(4)
plt.grid()
thickline, = plt.plot(sigmatheta_i, zs,color=lightgray,linestyle='-',linewidth=LW)
plt.plot(sigmatheta_m, zs,color=lightgray,linestyle='-',linewidth=LW)
plt.plot(sigmatheta_o, zs,color=lightgray,linestyle='-',linewidth=LW)
thinline, = plt.plot(sigmatheta_bi, invzs,color=black,linestyle='--',linewidth=0.5*LW)
plt.plot(sigmatheta_bm, invzs,color=black,linestyle='--',linewidth=0.5*LW)
plt.plot(sigmatheta_bo, invzs,color=black,linestyle='--',linewidth=0.5*LW)
abaqusin, = plt.plot(abaqus.sigmatheta_i, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='d',markersize=MS,markerfacecolor='white',markevery=4)
abaqusmid, = plt.plot(abaqus.sigmatheta_m, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='o',markersize=MS,markerfacecolor='white',markevery=4)
abaqusout, = plt.plot(abaqus.sigmatheta_o, abaqus.zinv,color=black,linestyle='none',linewidth=0.5*LW,marker='s',markersize=MS,markerfacecolor='white',markevery=4)
plt.gca().invert_yaxis()
plt.legend([thickline,thinline,abaqusin,abaqusmid,abaqusout],['Thick theory','Thin theory','ABAQUS FE (inner surf.)','ABAQUS FE (middle surf.)','ABAQUS FE (outer surf.)'],loc='center left', fontsize=FS, frameon=True, edgecolor=black)
plt.xlabel('$\sigma_\\theta$ [$N/mm^2$]',fontdict={'size':FS})
plt.ylabel('Axial coordinate $z$ [mm]',fontdict={'size':FS})

plt.show()