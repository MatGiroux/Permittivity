# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:01:29 2022

@author: girm0
"""

#this model is from https://doi.org/10.1117/1.JPE.10.025503 
#the interband absorption, Burstein-Moss shift and the fermi level has been calculated using the model in this paper https://doi.org/10.1016/0020-0891(80)90053-6

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
import sympy as sy


def Eg_0_T_calc(T,U): #function that computes the undoped bandgap

    #the following values are only for InAs
    Eg_0_0=0.417*U("eV") #undoped zero temperature bandgap
    delta=2.76E-4*U("eV/K")
    beta=93*U("K")
    #computing the undoped bandgap
    Eg_0_T=Eg_0_0-delta*T**2/(T+beta)
    
    return Eg_0_T


def mu_calc_ntype(N,T,U): #this function computes the electron mobility for n type InAs

    #these values are from Forcade 2022
    umin = 0.3*U("cm^2V^-1s^-1")#there is an error in Milovich 2020 we need to use cm^2 and not cm^-2
    umax = 30636.0*U("cm^2V^-1s^-1")
    Nref = 3.56E17*U("cm^-3")
    phi = 0.68*U("dimensionless")
    theta1 = 1.57*U("dimensionless")
    theta2 = 3.0*U("dimensionless")
    
    T_ref=300*U("K")
    
    #computing the electron mobility
    mu =  umin + (umax*(T_ref/T)**theta1-umin)/(1.0+(N/(Nref*(T/T_ref)**theta2))**phi)
    
    return mu


def mu_calc_ptype(N,T,U): #this function computes the electron mobility for p type InAs
    
    umin = 20.0*U("cm^2V^-1s^-1")#there is an error in Milovich 2020 we need to use cm^2 and not cm^-2
    umax = 530.0*U("cm^2V^-1s^-1")
    Nref = 1.1E17*U("cm^-3")
    phi = 0.46*U("dimensionless")
    theta1 = 2.3*U("dimensionless")
    theta2 = 3.0*U("dimensionless")
    
    T_ref=300*U("K")
    
    #computing the electron mobility
    mu =  umin + (umax*(T_ref/T)**theta1-umin)/(1.0+(N/(Nref*(T/T_ref)**theta2))**phi)
    
    return mu


def eps_imag_FCA_integrand_calc(X,zeta,X_TF): #this function computes the integrand for the calculation of the imaginary part of the permittivity due to the FCA
    
    #computing expressions to simplify the integrand expression
    expr_1=(np.sqrt(X+zeta)+np.sqrt(X))**2+X_TF
    expr_2=(np.sqrt(X+zeta)-np.sqrt(X))**2+X_TF
    
    #computing the final expression
    eps_FCA_integrand = 0.5*np.log(expr_1/expr_2)-(2.0*X_TF*np.sqrt(X*(X+zeta)))/(expr_1*expr_2)
    
    return eps_FCA_integrand


def eps_imag_FCA_calc(A,zeta,X_TF,U): #this is the implementation of equation 4.4a in Baltz 1972 https://doi.org/10.1002/pssb.2220510209

    #initiating an array for the loop
    eps_imag_FCA=[]

    #loop over the frequency
    for i in range(len(zeta)):
        #computing the integral
        if (1.0-zeta[i]).m_as("dimensionless")<0:
            eps_imag_FCA.append(A.m_as("dimensionless")*(zeta[i].m_as("dimensionless"))**(-4)*integrate.quadrature(eps_imag_FCA_integrand_calc, 0, 1, args=(zeta[i].m_as("dimensionless"),X_TF.m_as("dimensionless")), tol=1e-10, rtol=1e-6, maxiter=5000)[0])
        else:
            eps_imag_FCA.append(A.m_as("dimensionless")*(zeta[i].m_as("dimensionless"))**(-4)*integrate.quadrature(eps_imag_FCA_integrand_calc, (1.0-zeta[i]).m_as("dimensionless"), 1, args=(zeta[i].m_as("dimensionless"),X_TF.m_as("dimensionless")), tol=1e-10, rtol=1e-6, maxiter=5000)[0])
            
    return eps_imag_FCA*U("dimensionless")


def eps_FC_I_baltz_calc(w,E_F,eps_inf,epsilon0,e,N,hbar,m0,Eg_0_T,U): #this is the calculation of the imaginary part of the permittivity at high doping taken from Baltz 1972 https://doi.org/10.1002/pssb.2220510209

    #some varibales
    Z=1

    #computing q_TF
    q_TF=np.sqrt(3*N*e**2/(2*epsilon0*eps_inf*E_F))
    #computing the fermi wavevector
    k_F=np.sqrt(2*m0*E_F)/hbar
    #computing the number of impurities with charge Ze 
    R=Z*N
    #computing gamma
    gamma=R*(Z*e**2/(epsilon0*eps_inf))**2 #we removed K_F^4 here and in the A expression since they cancel each other
    #compute the A factor
    A=1/(12*U.pi**3)*e**2*gamma/(epsilon0*E_F**3)#we removed K_F^4 here and in the gamma expression since they cancel each other
    #computing zeta
    zeta=hbar*w/E_F
    #computing X_TF
    X_TF=(q_TF/k_F)**2
    
    #computing the imaginary part of the permittivity due to FCA
    eps_FC_I_2=eps_imag_FCA_calc(A,zeta,X_TF,U)
    
    return eps_FC_I_2


def eps_FC_I_calc(w,E_F,eps_inf,epsilon0,e,N,hbar,m0,wp_square,Eg_0_T,eps_Drude,U): #this model computes the imaginary part of the permittivity at high doping by combining the Baltz and Brude model depending on the region

    #splitting the array of the frequency in 2 at the plasmonic frequency
    if np.max(w)>=(wp_square)**0.5:
        index=np.where(w > (wp_square)**0.5)[0][0]
    else: #in the case where the frequency is never larger than the plasma frequency
        index=len(w)
        
    w_2=w[index:]
    
    #for the case where w<w_p
    eps_FC_I_1=eps_Drude[:index].imag
    
    #for the case where w>w_p
    eps_FC_I_2=eps_FC_I_baltz_calc(w_2,E_F,eps_inf,epsilon0,e,N,hbar,m0,Eg_0_T,U)
    
    #concatenating the two arrays
    eps_FC_I=np.concatenate((eps_FC_I_1,eps_FC_I_2))
    
    return eps_FC_I


def eps_FCL_calc(w,N,T,eps_inf,m0,mhh,mlh,N_doped,Eg_0_T,kb,hbar,P,U):#this function computes the free carrier and lattice response of the permittivity using a Drude-Lorentz model
    
    e=U.elementary_charge #charge of an electron
    epsilon0=U.vacuum_permittivity #permittivity of free space
    #for InAs #these values were taken from Forcade 2022
    g=9.23E11*U("rad/s")#phononic damping coefficient #from Adachi, "Optical Constants of...", 1999
    w_LO=4.55E13*U("rad/s")#frequency of the longitudinal optical phonon #0.0271 [eV] #from Adachi, "Properties of semiconductor and their alloys...", 2009
    w_TO=4.14E13*U("rad/s")#frequency of the tranverse optical phonon #0.0301 [eV] #from Adachi, "Properties of semiconductor and their alloys...", 2009

    #computing the electron mobility and effective mass
    if N_doped==True: #checking if the semiconductor is p or n type
        #computing the m_star and mu
        m_star=0.024*m0 #this is from S. Adachi, Handbook on Physical Properties of Semiconductors, 1st ed. (Springer, New York, 2017).
        mu=mu_calc_ntype(N,T,U)
        #computing the fermi level
        F=fermi_calc(N,Eg_0_T,kb,T,P,U)
        E_F=F-Eg_0_T
    else:
        #computing the m_star and mu
        m_star=(mhh**(3/2)+mlh**(3/2))**(2/3) #this is from S. Adachi, Handbook on Physical Properties of Semiconductors, 1st ed. (Springer, New York, 2017).
        mu=mu_calc_ptype(N,T,U)
        #setting some values to None if we have p-doped
        F=None
        E_F=None
    
    
    #computing the damping coefficient due to free carriers
    Gamma=e/(m_star*mu)
    
    #computing the plasma resonance
    wp_square=e**2*N/(eps_inf*epsilon0*m_star)
    
    #computing the lattice response of the permittivity using a Drude-Lorentz model
    eps_L=eps_inf*(1.0+(w_LO**2-w_TO**2)/(w_TO**2-w**2-1j*w*g))
    
    #computing the drude model typically used for the free carrier permittivity
    eps_Drude=eps_inf*(-wp_square/(w*(w+1j*Gamma)))
    
    #computing the free carrier response of the permittivity
    if N_doped==True: #if the sample is N-doped
        if E_F<kb*T:#for the low doping case
            eps_FC=eps_Drude
        else: #for the high doping we need a correct to the free carrier absorption
            eps_FC_R=eps_Drude.real #computing the real part of the permittivity
            eps_FC_I=eps_FC_I_calc(w,E_F,eps_inf,epsilon0,e,N,hbar,m0,wp_square,Eg_0_T,eps_Drude,U) #computing the imaginary part of the permittivity
            eps_FC=eps_FC_R+1j*eps_FC_I
    else:
        eps_FC=eps_Drude #if p-doped we simply use the Drude model
        
    #computing the total permittivity due to free carrier and lattice
    eps_FCL=eps_L+eps_FC
    
    return eps_FCL,F
    
    
    
def kw_calc(w,mhh,m0,Eg_0_T,hbar,P): #this function computes the wave vector
    
    lump = (1.0 + m0/mhh)*(2*hbar*w/Eg_0_T - 1.0)  
    k1 = (4.0/3.0*P**2 + hbar**2*Eg_0_T/m0*lump)/ (hbar**4/m0**2 * (1.0 + m0/mhh)**2)
    k2 = 1.0 - (1.0 - (4.0*hbar**4/(m0)**2*(1 + m0/mhh)**2*hbar*w*(hbar*w - Eg_0_T))/(4.0/3.0*P**2 + hbar**2*Eg_0_T/m0*lump)**2 )**(1/2)
    #here we set the array to complex type to perform the sqrt correctly
    kw=((k1*k2).astype(complex))**(1/2)
    
    return kw



def alpha_lh_calc(w,hbar,P,eps_inf,Eg_0_T):#this function computes the interband absorption for the light hole
    
    #here we set the array to complex type to perform the sqrt correctly
    sqr=(((hbar*w)**2-Eg_0_T**2).astype(complex))**(1/2)
    alpha_lh=(1.0+2*(Eg_0_T/(hbar*w))**2)*sqr/(137*(6*eps_inf)**(1/2)*4*P)
    
    return alpha_lh
    


def alpha_hh_calc(w,P,kw,Eg_0_T,hbar,m0,mhh,eps_inf):#this function computing the interband absorption for the heavy hole
    
    sqr = (1.0 + 8.0*P**2*kw**2/(3.0*Eg_0_T**2))**(1/2)
    top = Eg_0_T*kw/(hbar*w*2.0)*(sqr + 1.0) 
    bot = 137*eps_inf**(1/2)*(1.0 + 3.0/4.0*hbar**2*Eg_0_T*(1.0 + m0/mhh)*sqr/(m0*P**2))
    alpha_hh=top/bot
    
    return alpha_hh



def BM_hh_calc(w,hbar,kb,T,mhh,kw,F): #this function is used to compute the Burstein-Moss shift for heavy holes (taken from Anderson 1980)
    
    top = 1.0-np.exp(-w*hbar/(kb*T))
    bot = 1.0 + np.exp(- (F + hbar**2*kw**2/(2*mhh))/(kb*T))
    bot2 = 1.0 + np.exp(- (hbar*w - F - hbar**2*kw**2/(2*mhh))/(kb*T))
    BM=top/(bot*bot2)
    
    return BM



def BM_lh_calc(w,hbar,kb,T,Eg_0_T,F): #this function is used to compute the Burstein-Moss shift for light holes using Eq. 2 from Anderson 1980
    
    top = 1.0-np.exp(-w*hbar/(kb*T))
    bot = 1.0 + np.exp(- (hbar*w + Eg_0_T - 2*F)/(2*kb*T))
    bot2 = 1.0 + np.exp(- (hbar*w - Eg_0_T + 2*F)/(2*kb*T))
    BM=top/(bot*bot2)
    
    return BM


def Integral_calc(t,N_c,phi,N): #this function returns the solution to the integral that we are trying to solve

    x = sy.Symbol('x')
    y1 = x**(1/2)*(1+x/phi)**(1/2)*(1.0+2*x/phi)
    y2 = 1.0+sy.exp(x-t[0])
    y = y1/y2
    
    Integral=integrate.quad(lambda x_arg: y.subs(x, x_arg).evalf(), 0, np.inf)[0]
    
    diff=N/N_c-(2/((np.pi)**(1/2))*Integral)#computing the difference between the actual solution and the computed result
    
    return abs(diff) #returning the difference between the actual solution and the computed result
    

def fermi_calc(N,Eg_0_T,kb,T,P,U):#this function computes the fermi level (taken from Anderson 1980)
    
    N_c=2*(3*Eg_0_T*kb*T/(8*U.pi*P**2))**(3/2)
    phi=Eg_0_T/(kb*T)
    
    #we will find the solution to the integral by minimizing the difference between the right answer and the computed answer
    result = minimize(Integral_calc, method='SLSQP', x0=[1], args=(N_c.m_as("cm^(-3)"),phi.m_as("dimensionless"),N.m_as("cm^(-3)")))
    
    t=result.x[0]
    
    F=t*kb*T+Eg_0_T
    
    return F


def alpha_IB_calc(w,N,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,hbar,N_doped,kb,U):#this function computes the total Interband absorption coefficient

    #computing the wave vector
    kw=kw_calc(w,mhh,m0,Eg_0_T,hbar,P)

    #computing the light and heavy hole interband absorption
    alpha_lh=alpha_lh_calc(w,hbar,P,eps_inf,Eg_0_T)
    alpha_hh=alpha_hh_calc(w,P,kw,Eg_0_T,hbar,m0,mhh,eps_inf)
    
    if  N_doped == True:    #using moss-burstein effect when n_type
        #computing the light hole and heavy hole burstein moss shift
        BM_lh=BM_lh_calc(w,hbar,kb,T,Eg_0_T,F)
        BM_hh=BM_hh_calc(w,hbar,kb,T,mhh,kw,F)
    

    alpha_IB=[]
    
    for a in range(len(w)):
        if(hbar*w[a]<=Eg_0_T):
            #below bandgap, no absorption
            alpha_IB.append(0.0)
        else:
            if N_doped == True:    #using moss-burstein effect when n_type
                alpha_IB.append((alpha_lh[a]*BM_lh[a]+alpha_hh[a]*BM_hh[a]).m_as("rad/m"))
            else: #p-type material, thus dont include moss-burstein effect
                alpha_IB.append((alpha_lh[a]+ alpha_hh[a]).m_as("rad/m"))
    
    return alpha_IB*U("rad/m")



def eps_IB_calc(w,N,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,hbar,N_doped,kb,U):#this function computes the Interband response of the permittivity
    

    #computing the total Interband absorption coefficient
    alpha_IB=alpha_IB_calc(w,N,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,hbar,N_doped,kb,U)

    
    m_IB_p=eps_inf**(1/2) #this is an approximation to the Kramers-Kronig relation taken from Forcade 2022
    m_IB_pp=alpha_IB*U.c/(2*w)
    
    eps_IB=(m_IB_p**2-m_IB_pp**2)+2j*m_IB_p*m_IB_pp
    
    return eps_IB
    


def tot_eps_calc(w,N,T,mhh,mlh,P,eps_inf,m0,hbar,N_doped,kb,U):#this function computes the sum of the lattice, free carrier and Interband responses of the permittivty and returns the total permittivity
    
    
    Eg_0_T=Eg_0_T_calc(T,U) #computing the undoped bandgap

    eps_FCL,F=eps_FCL_calc(w,N,T,eps_inf,m0,mhh,mlh,N_doped,Eg_0_T,kb,hbar,P,U)#computing the free carrier response of the permittivity
    
    eps_IB=eps_IB_calc(w,N,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,hbar,N_doped,kb,U)
    
    #computing the total permittivity
    eps=eps_FCL+eps_IB-eps_inf
    
    return eps
    



def epsilon_InAs(lambd,N,T,N_doped,U):

    m0=U.electron_mass #mass of an electron
    hbar=U.hbar #dirac constant (planck constant)
    kb=U.k#boltzmann constant

    #for InAs
    eps_inf=11.6*U("dimensionless")#background permittivity #from p.738 of Adachi, S. (2017). III-V Ternary and Quaternary Compounds. In: Kasap, S., Capper, P. (eds) Springer Handbook of Electronic and Photonic Materials. Springer Handbooks. Springer, Cham. https://doi.org/10.1007/978-3-319-48933-9_30
    #these variable are for the interband absorption
    mhh=0.57*m0 #this is from Piprek “Semiconductor optoelectronic devices: Introduction to physics and simulations in the Table 1.1 (p.7)). 
    mlh=0.026*m0 #this is from Milovich 2020
    P=9.05E-8*U("eV*cm") #the value 9.05E-8*U("eV*cm") was calculated from Vurgaftman 2001 https://doi.org/10.1063/1.1368156 
        
    #computing the frequency in rad/s
    w=2*U.pi*U.c/(lambd)
    
    
    #computing the total permittivity of InAs
    eps_InAs=tot_eps_calc(w,N,T,mhh,mlh,P,eps_inf,m0,hbar,N_doped,kb,U)

    return eps_InAs
    
