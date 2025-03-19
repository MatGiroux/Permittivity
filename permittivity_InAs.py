# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:01:29 2022

@author: girm0
"""

#The main user-facing function is epsilon_InAs. This function computes the dielectric function for InAs.

#this model is adapted from 
#D. Milovich et al., “Design of an indium arsenide cell for near-field thermophotovoltaic devices,” J. Photon. Energy, vol. 10, no. 02, p. 1, Jun. 2020, doi: 10.1117/1.JPE.10.025503.

#the interband absorption, Burstein-Moss shift and the fermi level has been calculated using the model from
#W. W. Anderson, “Absorption constant of Pb1−xSnxTe and Hg1−xCdxTe alloys,” Infrared Physics, vol. 20, no. 6, pp. 363–372, Nov. 1980, doi: 10.1016/0020-0891(80)90053-6.

#to correct the overestimation of the free carrier absorption we incorporate an ionized impurity scattering model from
#R. Von Baltz and W. Escher, “Quantum Theory of Free Carrier Absorption,” Physica Status Solidi (b), vol. 51, no. 2, pp. 499–507, Jun. 1972, doi: 10.1002/pssb.2220510209.

#we also employ the updated parameters and the approximation to the Kramers-Kronig relations for the interband term from
#G. Forcade et al., “Efficiency-optimized near-field thermophotovoltaics using InAs and InAsSbP”, Applied Physics Letters, vol. 121, no. 19, p.193903, Nov. 2022, doi: 10.1063/5.0116806.

#we employ the carrier concentration model described in chapter 4 of
#S. M. Sze, "Physics of semiconductor devices", John Wiley & Sons, 1969.

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
import sympy as sy
from scipy.fftpack import hilbert
from scipy.interpolate import interp1d



def epsilon_InAs(w,Nd,T,N_doped,model_choice,N_calc_choice,U):
    '''This function computes the permittivity of InAs.
        w:          frequency
        Nd:         doping concentration
        T:          temperature 
        N_doped:    boolean. True if the semiconductor is n-type and False if it is p-type
        model_choice: "Approximation" or "Kramers-Kronig"
        N_calc_choice: True to calculate the carrier concentration including thermal generation or False to approximate it to Nd
    
    '''

    m0=U.electron_mass #mass of an electron

    #for InAs
    eps_inf=11.6*U("dimensionless")#background permittivity #from p.738 of Adachi, S. (2017). III-V Ternary and Quaternary Compounds. In: Kasap, S., Capper, P. (eds) Springer Handbook of Electronic and Photonic Materials. Springer Handbooks. Springer, Cham. https://doi.org/10.1007/978-3-319-48933-9_30
    #these variable are for the interband absorption
    mhh=0.57*m0 #this is from Piprek “Semiconductor optoelectronic devices: Introduction to physics and simulations in the Table 1.1 (p.7)). 
    mlh=0.026*m0 #this is from Milovich 2020
    P=9.05E-8*U("eV*cm") #Calculated from I. Vurgaftman, J. R. Meyer, and L. R. Ram-Mohan, “Band parameters for III–V compound semiconductors and their alloys,” Journal of Applied Physics, vol. 89, no. 11, pp. 5815–5875, Jun. 2001, doi: 10.1063/1.1368156.
    
    
    #computing the total permittivity of InAs
    eps_InAs=tot_eps_calc(w,Nd,T,mhh,mlh,P,eps_inf,m0,N_doped,model_choice,N_calc_choice,U)

    return eps_InAs


def Eg_0_T_calc(T,U): 
    '''This function computes the undoped bandgap.
    '''

    #the following values are only for InAs
    #these values are taken from I. Vurgaftman, J. R. Meyer and L. R. Ram-Mohan, "Band parameters for III–V compound semiconductors and their alloys," Journal of Applied Physics, vol. 89, p. 5815–5875, 2001. 
    Eg_0_0=0.417*U("eV") #undoped zero temperature bandgap
    delta=2.76E-4*U("eV/K")
    beta=93*U("K")
    #computing the undoped bandgap
    Eg_0_T=Eg_0_0-delta*T**2/(T+beta)
    
    return Eg_0_T


def N_calc(Nd,m0,mhh,mlh,T,Eg_0_T,U): 
    '''This function computes the thermally excited free electrons and holes.
    
    The model implemented here is taken from chapter 4 of Sze 1970.
    '''
    
    #computing the electron and hole effective mass
    me=0.024*m0 #this is from S. Adachi, Handbook on physical properties of semiconductors, Springer Science & Business Media, 2004. 
    mh=(mhh**(3/2)+mlh**(3/2))**(2/3) #this is from S. Adachi, Handbook on physical properties of semiconductors, Springer Science & Business Media, 2004. 

    N_c=2*((2*np.pi*me*U.k*T/(U.h**2))**(3/2)) #effective density of state in the conduction band
    N_v=2*((2*np.pi*mh*U.k*T/(U.h**2))**(3/2)) #effective density of state in the valence band
    
    N_th=np.sqrt(N_c*N_v*np.exp(-Eg_0_T/(U.k*T))) #computing the tehrmally excited free electrons and holes

    #computing the carrier concentration
    N=1/2*(Nd+np.sqrt((Nd)**2+4*N_th**2))
    
    return N


def mu_calc_ntype(Nd,T,U): 
    '''This function computes the electron mobility for n type InAs.
    '''

    #these values are from Forcade 2022
    umin = 0.3*U("cm^2V^-1s^-1")#there is an error in Milovich 2020 we need to use cm^2 and not cm^-2
    umax = 30636.0*U("cm^2V^-1s^-1")
    Nref = 3.56E17*U("cm^-3")
    phi = 0.68*U("dimensionless")
    theta1 = 1.57*U("dimensionless")
    theta2 = 3.0*U("dimensionless")
    
    T_ref=300*U("K")
    
    #computing the electron mobility
    mu =  umin + (umax*(T_ref/T)**theta1-umin)/(1.0+(Nd/(Nref*(T/T_ref)**theta2))**phi)
    
    return mu


def mu_calc_ptype(Nd,T,U): 
    '''This function computes the electron mobility for p type InAs.
    '''
    
    #these values are from Milovich 2020
    umin = 20.0*U("cm^2V^-1s^-1")#there is an error in Milovich 2020 we need to use cm^2 and not cm^-2
    umax = 530.0*U("cm^2V^-1s^-1")
    Nref = 1.1E17*U("cm^-3")
    phi = 0.46*U("dimensionless")
    theta1 = 2.3*U("dimensionless")
    theta2 = 3.0*U("dimensionless")
    
    T_ref=300*U("K")
    
    #computing the electron mobility
    mu =  umin + (umax*(T_ref/T)**theta1-umin)/(1.0+(Nd/(Nref*(T/T_ref)**theta2))**phi)
    
    return mu


def eps_imag_FCA_integrand_calc(X,zeta,X_TF): 
    '''this function computes the integrand for the calculation of the imaginary 
    part of the permittivity due to the free-carrier absorption (FCA).
    '''
    
    #computing expressions to simplify the integrand expression
    expr_1=(np.sqrt(X+zeta)+np.sqrt(X))**2+X_TF
    expr_2=(np.sqrt(X+zeta)-np.sqrt(X))**2+X_TF
    
    #computing the final expression
    eps_FCA_integrand = 0.5*np.log(expr_1/expr_2)-(2.0*X_TF*np.sqrt(X*(X+zeta)))/(expr_1*expr_2)
    
    return eps_FCA_integrand


def eps_imag_FCA_calc(A,zeta,X_TF,U): 
    '''This function compute the imaginary part of the permittivity due to 
    free-carrier absorption according to equation 4.4a of Baltz 1972.
    '''

    #initiating an array for the loop
    eps_imag_FCA=np.zeros(len(zeta))

    #loop over the frequency
    for i in range(len(zeta)):
        #computing the integral
        if (1.0-zeta[i]).m_as("dimensionless")<0:
            eps_imag_FCA[i]=A.m_as("dimensionless")*(zeta[i].m_as("dimensionless"))**(-4)*integrate.quadrature(eps_imag_FCA_integrand_calc, 0, 1, args=(zeta[i].m_as("dimensionless"),X_TF.m_as("dimensionless")), tol=1e-10, rtol=1e-6, maxiter=5000)[0]
        else:
            eps_imag_FCA[i]=A.m_as("dimensionless")*(zeta[i].m_as("dimensionless"))**(-4)*integrate.quadrature(eps_imag_FCA_integrand_calc, (1.0-zeta[i]).m_as("dimensionless"), 1, args=(zeta[i].m_as("dimensionless"),X_TF.m_as("dimensionless")), tol=1e-10, rtol=1e-6, maxiter=5000)[0]
            
    return eps_imag_FCA*U("dimensionless")


def eps_FC_I_baltz_calc(w,E_F,eps_inf,epsilon0,e,Nd,N,m_star,U): 
    '''This function computes the imaginary part of the free-carrier permittivity
    contribution at high doping.
    
    The model implemented here is taken from Baltz 1972.
    '''

    #some varibales
    Z=1

    #computing q_TF
    q_TF=np.sqrt(3*N*e**2/(2*epsilon0*eps_inf*E_F))
    #computing the fermi wavevector
    k_F=np.sqrt(2*m_star*E_F)/U.hbar
    #computing the number of impurities with charge Ze 
    R=Z*Nd
    #computing gamma
    gamma=R*(Z*e**2/(epsilon0*eps_inf))**2 #we removed K_F^4 here and in the A expression since they cancel each other
    #compute the A factor
    A=1/(12*U.pi**3)*e**2*gamma/(epsilon0*E_F**3)#we removed K_F^4 here and in the gamma expression since they cancel each other
    #computing zeta
    zeta=U.hbar*w/E_F
    #computing X_TF
    X_TF=(q_TF/k_F)**2
    
    #computing the imaginary part of the permittivity due to FCA
    eps_FC_I_2=eps_imag_FCA_calc(A,zeta,X_TF,U)
    
    return eps_FC_I_2


def eps_FC_I_calc(w,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,eps_Drude,U): 
    '''This function computes the imaginary part of the free-carrier permittivity 
    contribution at high doping by combining the model from Baltz 1972 and the 
    Drude model depending on the region.
    '''

    #splitting the array of the frequency in 2 at the plasmonic frequency
    if np.max(w)>=(wp_square)**0.5:
        index=np.where(w > (wp_square)**0.5)[0][0]
    else: #in the case where the frequency is never larger than the plasma frequency
        index=len(w)
        
    w_2=w[index:]
    
    #for the case where w<w_p
    eps_FC_I_1=eps_Drude[:index].imag
    
    #for the case where w>w_p
    eps_FC_I_2=eps_FC_I_baltz_calc(w_2,E_F,eps_inf,epsilon0,e,Nd,N,m_star,U)
    
    #concatenating the two arrays
    eps_FC_I=np.concatenate((eps_FC_I_1,eps_FC_I_2))
    
    return eps_FC_I

def Kramers_Kronig_IB(w,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,U): 
    '''This function performs the Kramers Kronig relations to find the real part 
    of the interband refractive index given the imaginary part of the refractive 
    index, which is itself determined from the interband absorption coefficient.
    '''

    #Generating an array for the frequency
    len_w=16000 #length of the array
    E_max=6*U("eV") #we want to up to at least 6 eV to make sure we include all the high energy contributions
    w_max=max(max(w),E_max/U.hbar) #finding the max between the minimum max value for the computation and the max value of the array
    w_vec_KK=np.linspace(-w_max,w_max,len_w*2)[len_w:]# we symmetrize the frequency and then crop all the negative values
    
    #computing the total Interband absorption coefficient
    alpha_IB_KK=alpha_IB_calc(w_vec_KK,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,U)
    #computing the refractive index from the absorption coefficient
    m_IB_pp_KK=alpha_IB_KK*U.c/(2*w_vec_KK)
    
    #computing the Hilbert transform
    Hilbert_m_IB_KK=Hilbert_transform_calc(w_vec_KK,m_IB_pp_KK.m_as("dimensionless"))#computing the Hilbert transform 
    #adding eps_inf component required for the Hilbert transform
    m_IB_p_KK=eps_inf**(1/2)+Hilbert_m_IB_KK
    
    #Interpolating to get the values for the w array
    interp_function = interp1d(w_vec_KK.m_as("rad/s"), m_IB_p_KK.m_as("dimensionless"), kind='linear')
    m_IB_p = interp_function(w.m_as("rad/s"))
    
    return m_IB_p


def Kramers_Kronig_FC(w,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,Gamma,U): 
    '''This function performs the Kramers Kronig relations to find the real part
    of the free carrier permittivity given the imaginary part of the permittivity, 
    which is itself determined using the model from Baltz 1972.
    '''

    #Generating an array for the frequency (we generate a new array to have symetrical array that goes up to large energy)
    len_w=16000 #length of the array
    E_max=6*U("eV") #we want to up to at least 6 eV to make sure we include all the high energy contributions
    w_max=max(max(w),E_max/U.hbar) #finding the max between the minimum max value for the computation and the max value of the array
    w_vec_KK=np.linspace(-w_max,w_max,len_w*2)[len_w:]# we symmetrize the frequency and then crop all the negative values
    
    #computing the Drude model for the new frequency array
    eps_Drude_KK=eps_inf*(1.0-wp_square/(w_vec_KK*(w_vec_KK+1j*Gamma)))
    #computing the imaginary part of the permittivity
    eps_FC_I_KK=eps_FC_I_calc(w_vec_KK,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,eps_Drude_KK,U)
    
    #computing the Hilbert transform
    Hilbert_eps_FC_KK=Hilbert_transform_calc(w_vec_KK,(eps_FC_I_KK-eps_Drude_KK.imag).m_as("dimensionless"))
    #adding the real part of the Drude model to the solution
    eps_FC_R_KK=Hilbert_eps_FC_KK+eps_Drude_KK.real #here we do not add eps_inf as it is already included in the Lorentz model
    
    #Interpolating to get the values for the w array
    interp_function = interp1d(w_vec_KK.m_as("rad/s"), eps_FC_R_KK.m_as("dimensionless"), kind='linear')
    eps_FC_R = interp_function(w.m_as("rad/s"))
    
    return eps_FC_R
        

def Hilbert_transform_calc(w_vec_KK,chi_I): 
    '''This function computes the Hilbert transform of a data set.
    It extends the function to negative frequency by assuming the function is odd; it is 
    thus appropriate for converting the imaginary part of epsilon to the real part.
    For the K-K relation going from real to imaginary, the function should be even.
    '''
    
    #Symmetrize the data to extend to negative values
    chi_I_full = np.concatenate((-chi_I[::-1], chi_I)) #assuming an odd function
    
    #computing the real of the interband refractive index using the Kramers-Kronig relation
    Hilbert_R=hilbert(chi_I_full)[len(w_vec_KK):]# Skip the negative part
    
    return Hilbert_R


def eps_FC_R_calc(w,eps_Drude,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,Gamma,model_choice,U): 
    '''This model computes the real part of the free-carrier permittivity contribution.
    
    The flag model_choice = "Approximation" or "Kramers-Kronig" allows the user to choose between computing the Kramers-Kronig 
    relations to find the real part of the permittivity or simply approximating 
    it to the real part of the Drude model.
    '''

    #computing the real part of the permittivity
    if model_choice=="Approximation":#if the user chose the approximation to the Kramers-Kronig relation
        eps_FC_R=eps_Drude.real #approximating to the Drude model
    elif model_choice=="Kramers-Kronig": #if the user chose to compute the Kramers-Kronig relations
        eps_FC_R=Kramers_Kronig_FC(w,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,Gamma,U) #computing the real part of the permittivity using the Kramers-Kronig relations
        
    return eps_FC_R


def eps_FC_calc(w,Nd,T,eps_inf,m0,mhh,mlh,N_doped,Eg_0_T,P,model_choice,N_calc_choice,U):
    '''This function computes the free carrier contribution of the permittivity
    using a Drude model or a combination of the Drude model with the model from 
    Baltz 1972 depending on the doping and temperature of the sample.

    
    N_calc_choice can be True to calculate the carrier concentration including thermal generation or False to approximate it to the doping level.


    '''
    
    e=U.elementary_charge #charge of an electron
    epsilon0=U.vacuum_permittivity #permittivity of free space
    
    #computing the carrier concentration from the doping level
    if N_calc_choice==True:#if the user chose to compute the carrier concentration
        N=N_calc(Nd,m0,mhh,mlh,T,Eg_0_T,U) #this computes the carrier concentration by accounting for thermally excited carriers
    elif N_calc_choice==False: #if the user chose to approximate the carrier concentration to the doping level
        N=Nd #setting the carrier concentration to the doping level

    #computing the electron mobility and effective mass
    if N_doped==True: #checking if the semiconductor is p or n type
        #computing the m_star and mu
        m_star=0.024*m0 #this is from S. Adachi, Handbook on physical properties of semiconductors, Springer Science & Business Media, 2004. 
        mu=mu_calc_ntype(Nd,T,U)
        #computing the fermi level
        F=fermi_calc(N,Eg_0_T,T,P,U)
        E_F=F-Eg_0_T
    else:
        #computing the m_star and mu
        m_star=(mhh**(3/2)+mlh**(3/2))**(2/3) #this is from S. Adachi, Handbook on physical properties of semiconductors, Springer Science & Business Media, 2004. 
        mu=mu_calc_ptype(Nd,T,U)
        #setting some values to None if we have p-doped
        F=None
        E_F=None
    
    #computing the damping coefficient due to free carriers
    Gamma=e/(m_star*mu)
    
    #computing the plasma resonance
    wp_square=e**2*N/(eps_inf*epsilon0*m_star)
    
    #computing the drude model typically used for the free carrier permittivity
    eps_Drude=eps_inf*(1.0-wp_square/(w*(w+1j*Gamma)))
    
    #computing the free carrier response of the permittivity
    if N_doped==True: #if the sample is N-doped
        if E_F<U.k*T:#for the low doping case
            eps_FC=eps_Drude
        else: #for the high doping we need a correct to the free carrier absorption
            eps_FC_I=eps_FC_I_calc(w,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,eps_Drude,U) #computing the imaginary part of the permittivity
            eps_FC_R=eps_FC_R_calc(w,eps_Drude,E_F,eps_inf,epsilon0,e,Nd,N,m_star,wp_square,Gamma,model_choice,U) #computing the real part of the permittivity
            eps_FC=eps_FC_R+1j*eps_FC_I
    else:
        eps_FC=eps_Drude #if p-doped we simply use the Drude model
    
    return eps_FC,F

    
def eps_Lattice_calc(w,eps_inf,U):
    '''This function computes the lattice response contribution of the
    permittivity using a Lorentz model.
    '''
    
    #for InAs #these values were taken from Forcade 2022
    g=9.23E11*U("rad/s")#phononic damping coefficient #from S. Adachi, Optical Properties of Crystalline and Amorphous Semiconductors: Materials and Fundamental Principles. New York, NY: Springer, 1999.
    w_LO=4.55E13*U("rad/s")#frequency of the longitudinal optical phonon #from S. Adachi, Properties of semiconductor alloys: group-IV, III-V and II-VI semiconductors. John Wiley & Sons, 2009.
    w_TO=4.14E13*U("rad/s")#frequency of the tranverse optical phonon #from S. Adachi, Properties of semiconductor alloys: group-IV, III-V and II-VI semiconductors. John Wiley & Sons, 2009.
    
    #computing the lattice response of the permittivity using a Lorentz model
    eps_Lattice=eps_inf*(1.0+(w_LO**2-w_TO**2)/(w_TO**2-w**2-1j*w*g))
    
    return eps_Lattice

    
def kw_calc(w,mhh,m0,Eg_0_T,P,U): 
    '''This function computes the wave vector.
    '''
    
    lump = (1.0 + m0/mhh)*(2*U.hbar*w/Eg_0_T - 1.0)  
    k1 = (4.0/3.0*P**2 + U.hbar**2*Eg_0_T/m0*lump)/ (U.hbar**4/m0**2 * (1.0 + m0/mhh)**2)
    k2 = 1.0 - (1.0 - (4.0*U.hbar**4/(m0)**2*(1 + m0/mhh)**2*U.hbar*w*(U.hbar*w - Eg_0_T))/(4.0/3.0*P**2 + U.hbar**2*Eg_0_T/m0*lump)**2 )**(1/2)
    #here we set the array to complex type to perform the sqrt correctly
    kw=((k1*k2).astype(complex))**(1/2)
    
    return kw



def alpha_lh_calc(w,P,eps_inf,Eg_0_T,U):
    '''This function computes the interband absorption for light holes.
    '''
    
    #here we set the array to complex type to perform the sqrt correctly
    sqr=(((U.hbar*w)**2-Eg_0_T**2).astype(complex))**(1/2)
    alpha_lh=(1.0+2*(Eg_0_T/(U.hbar*w))**2)*sqr/(137*(6*eps_inf)**(1/2)*4*P)
    
    return alpha_lh
    


def alpha_hh_calc(w,P,kw,Eg_0_T,m0,mhh,eps_inf,U):
    '''This function computing the interband absorption for heavy holes.
    '''
    
    sqr = (1.0 + 8.0*P**2*kw**2/(3.0*Eg_0_T**2))**(1/2)
    top = Eg_0_T*kw/(U.hbar*w*2.0)*(sqr + 1.0) 
    bot = 137*eps_inf**(1/2)*(1.0 + 3.0/4.0*U.hbar**2*Eg_0_T*(1.0 + m0/mhh)*sqr/(m0*P**2))
    alpha_hh=top/bot
    
    return alpha_hh



def BM_hh_calc(w,T,mhh,kw,F,U): 
    '''This function is used to compute the Burstein-Moss shift for heavy holes
    (taken from Anderson 1980).
    '''
    
    top = 1.0-np.exp(-w*U.hbar/(U.k*T))
    bot = 1.0 + np.exp(- (F + U.hbar**2*kw**2/(2*mhh))/(U.k*T))
    bot2 = 1.0 + np.exp(- (U.hbar*w - F - U.hbar**2*kw**2/(2*mhh))/(U.k*T))
    BM_hh=top/(bot*bot2)
    
    return BM_hh



def BM_lh_calc(w,T,Eg_0_T,F,U): 
    '''This function is used to compute the Burstein-Moss shift for light holes
    using Eq. 2 from Anderson 1980.
    '''
    
    top = 1.0-np.exp(-w*U.hbar/(U.k*T))
    bot = 1.0 + np.exp(- (U.hbar*w + Eg_0_T - 2*F)/(2*U.k*T))
    bot2 = 1.0 + np.exp(- (U.hbar*w - Eg_0_T + 2*F)/(2*U.k*T))
    BM_lh=top/(bot*bot2)
    
    return BM_lh


def Integral_calc(t,N_c,phi,N): 
    '''This function returns the solution to the integral that we are trying to
    solve to compute the Fermi level.
    '''

    x = sy.Symbol('x')
    y1 = x**(1/2)*(1+x/phi)**(1/2)*(1.0+2*x/phi)
    y2 = 1.0+sy.exp(x-t[0])
    y = y1/y2
    
    Integral=integrate.quad(lambda x_arg: y.subs(x, x_arg).evalf(), 0, np.inf)[0]
    
    diff=N/N_c-(2/((np.pi)**(1/2))*Integral)#computing the difference between the actual solution and the computed result
    
    return abs(diff) #returning the difference between the actual solution and the computed result
    

def fermi_calc(N,Eg_0_T,T,P,U):
    '''This function computes the Fermi level (taken from Anderson 1980).
    '''
    
    N_c=2*(3*Eg_0_T*U.k*T/(8*U.pi*P**2))**(3/2)
    phi=Eg_0_T/(U.k*T)
    
    #we will find the solution to the integral by minimizing the difference between the right answer and the computed answer
    result = minimize(Integral_calc, method='SLSQP', x0=[1], args=(N_c.m_as("cm^(-3)"),phi.m_as("dimensionless"),N.m_as("cm^(-3)")))
    
    t=result.x[0]
    
    F=t*U.k*T+Eg_0_T
    
    return F

def f_func(x): 
    '''Subfunction for the smooth cutoff function employed for the light hole 
    absorption calculations.
    '''
    
    #exponential function
    f=np.exp(-1/x)

    #removing the exponential values before the cutoff
    f[x<=0]=0.0

    return f


def g_func(x): 
    '''Subfunction for the smooth cutoff function the light hole 
    absorption calculations.
    '''

    return f_func(x)/(f_func(x)+f_func(1-x))


def cutoff_func(w,U): 
    '''Function for the smooth cutoff employed for the light hole 
    absorption calculations.
    '''

    #converting the frequency to energy
    E_vec=w*U.hbar
    
    #parameters for the cutoff function
    E_cutoff=0.75*U("eV") #we chose a cutoff energy of 0.75 since it is the energy at which split-off hole absorption would begin to contribute to alpha(w)
    delta=1*U("eV") #range of the cutoff
    
    #computing the smooth cutoff function
    h=1-g_func((E_vec-E_cutoff)/delta)

    return h


def alpha_IB_calc(w,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,U):
    '''This function computes the interband absorption coefficient. 
    
    Here we employ a smooth cutoff of the light hole absorption since it comes 
    from k.p theory that is valid only for w near the bandgap. This form continues 
    to grow at large w, which makes the K-K relations non-convergent. We apply a 
    smooth cutoff begining at 0.75 eV, which is the energy at which split-off 
    hole absorption would begin to contribute to alpha(w). We apply a cutoff 
    over a range of 1 eV.
    '''

    #computing the wave vector
    kw=kw_calc(w,mhh,m0,Eg_0_T,P,U)

    #computing the light and heavy hole interband absorption
    alpha_lh=alpha_lh_calc(w,P,eps_inf,Eg_0_T,U)
    alpha_hh=alpha_hh_calc(w,P,kw,Eg_0_T,m0,mhh,eps_inf,U)
    
    if  N_doped == True:    #using moss-burstein effect when n_type
        #computing the light hole and heavy hole burstein moss shift
        BM_lh=BM_lh_calc(w,T,Eg_0_T,F,U)
        BM_hh=BM_hh_calc(w,T,mhh,kw,F,U)
    
    #computing the smooth cutoff function
    h=cutoff_func(w,U) #we implement a cutoff on alpha_lh since it comes from k.p theory that is valid only for w near the bandgap. This form continues to grow at large w, which makes the K-K relations non-convergent. We apply a smooth cutoff begining at 0.75 eV, which is the energy at which split-off hole absorption would begin to contribute to alpha(w). We apply a cutoff over a range of 1 eV

    #computing the interband absorption
    if N_doped == True:    #using moss-burstein effect when n_type
        alpha_IB=(alpha_lh*BM_lh*h+alpha_hh*BM_hh).m_as("rad/m")
    else: #p-type material, thus dont include moss-burstein effect
        alpha_IB=(alpha_lh*h+ alpha_hh).m_as("rad/m")
        
    #set to zero the below bandgap absorption
    alpha_IB[U.hbar*w<=Eg_0_T]=0.0
    
    return alpha_IB*U("rad/m")



def eps_IB_calc(w,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,model_choice,U):
    '''This function computes the interband contribution to the permittivity.
    
    Here a flag allows the user to choose between computing the Kramers-Kronig 
    relations to find the real part of the refractive index or simply using the 
    approximation from Forcade 2022.
    '''
    

    #computing the total Interband absorption coefficient
    alpha_IB=alpha_IB_calc(w,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,U)

    #computing the refractive index from the absorption coefficient
    m_IB_pp=alpha_IB*U.c/(2*w)
    
    #computing the real part of the refractive index
    if model_choice=="Approximation":#if the user chose the approximation to the Kramers-Kronig relation
        m_IB_p=eps_inf**(1/2) #this is an approximation to the Kramers-Kronig relation taken from Forcade 2022
    elif model_choice=="Kramers-Kronig": #if the user chose to compute the Kramers-Kronig relations
        m_IB_p=Kramers_Kronig_IB(w,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,U) #computing the real part of the refractive index using the Kramers-Kronig relations
        
    #computing the permittivity from the refractive index
    eps_IB=(m_IB_p**2-m_IB_pp**2)+2j*m_IB_p*m_IB_pp
    
    return eps_IB
    


def tot_eps_calc(w,Nd,T,mhh,mlh,P,eps_inf,m0,N_doped,model_choice,N_calc_choice,U):
    '''This function computes the sum of the lattice, free carrier and interband 
    contributions to the permittivty and returns the total permittivity.
    '''
    
    Eg_0_T=Eg_0_T_calc(T,U) #computing the undoped bandgap

    #computing the permittivity from each contribution
    eps_FC,F=eps_FC_calc(w,Nd,T,eps_inf,m0,mhh,mlh,N_doped,Eg_0_T,P,model_choice,N_calc_choice,U)#computing the free carrier response of the permittivity
    eps_Lattice=eps_Lattice_calc(w,eps_inf,U) #computing the lattice response of the permittivity
    eps_IB=eps_IB_calc(w,T,eps_inf,mhh,mlh,m0,P,Eg_0_T,F,N_doped,model_choice,U) #computing the interband response of the permittivity
    
    #computing the susceptibility from each contribution
    chi_FC=eps_FC-eps_inf
    chi_Lattice=eps_Lattice-eps_inf
    chi_IB=eps_IB-eps_inf
    
    #computing the total permittivity
    eps_tot=eps_inf+chi_FC+chi_Lattice+chi_IB
    
    return eps_tot


    
