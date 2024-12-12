# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:32:38 2023

@author: Mathieu Giroux
"""

#This model is a combination of the models presented in: Fu 2005, Basu 2010, Lee 2005 and Timans 1993

#C.J. Fu 2005 J of Heat and Mass Transfer Nanoscale radiation heat transfer for silicon at different doping levels
#Basu, Lee, Zhang. "Infrared radiative properties of heavily doped silicon at room temperature", 2010.
#Lee, B. J., Zhang, Z. M., Early, E. A., DeWitt, D. P., &  Tsai, B. K. (2005). Modeling radiative properties of silicon with coatings and comparison with reflectance measurements. Journal of thermophysics and heat transfer, 19(4), 558-565. 
#Timans, P. J. (1993) Emissivity of silicon at elevatef temperatures. Journal of Applied Physics. doi: https://doi.org/10.1063/1.355159


import numpy as np
import math


def tau_calc(U,el,me,mh,T,T_inf,ND,NA): 
    '''This function computes the scattering time of electrons and holes. 
    
    This model is from Fu 2005 except for the room temperature scattering rates 
    which are from Basu 2010
    '''
        
    #computing the mobility of electrons and holes
    mu_e=mu_e_calc(U,ND)
    mu_h=mu_h_calc(U,NA)

    #computing the room temperature electron and hole scattering time (from Basu 2010)
    tau_e_0=(el/(me*mu_e))**-1
    tau_h_0=(el/(mh*mu_h))**-1
    
    #room temperature electron-lattice and hole-lattice scattering time (from Fu 2005)
    tau_el_0=2.23E-13*U("s") 
    tau_hl_0=1.06E-13*U("s")
    
    #computing the rooom temperature electron-impurity and hole-impurity scattering times (from Fu 2005)
    tau_ed_0=tau_e_0*tau_el_0/(tau_el_0-tau_e_0)
    tau_hd_0=tau_h_0*tau_hl_0/(tau_hl_0-tau_h_0)
    
    #computing the temperature dependant scattering times (from Fu 2005)
    tau_ed=tau_ed_0*(T/T_inf)**1.5
    tau_hd=tau_hd_0*(T/T_inf)**1.5
    tau_el=tau_el_0*(T/T_inf)**(-3.8)
    tau_hl=tau_hl_0*(T/T_inf)**(-3.6)
    
    #computing the total scattering time (from Fu 2005)
    tau_e=(1/tau_el+1/tau_ed)**-1
    tau_h=(1/tau_hl+1/tau_hd)**-1
    
    return tau_e,tau_h
    

def mu_e_calc(U,ND): 
    '''This function computes the mobility of electrons (this is from Basu 2010).
    '''
    
    #variables taken from Basu 2010
    mu_1=68.5*U("cm^2/V/s")
    mu_2=56.1*U("cm^2/V/s")
    mu_max=1414*U("cm^2/V/s")
    Cr=9.2E16*U("cm^(-3)")
    Cs=3.41E20*U("cm^(-3)")
    alpha=0.711*U("dimensionless")
    beta=1.98*U("dimensionless")
    
    #computing the mobility of electrons
    mu_e=mu_1+(mu_max-mu_1)/(1+(ND/Cr)**alpha)-mu_2*((ND/Cs)**beta)/((ND/Cs)**beta+1) #we modified the last part of the equation to allow to compute when ND=0
    
    return mu_e


def mu_h_calc(U,NA): 
    '''This function computes the mobility of holes (this is from Basu 2010).
    '''
    
    #variables taken from Basu 2010
    mu_1=44.9*U("cm^2/V/s")
    mu_2=29.0*U("cm^2/V/s")
    mu_max=470.5*U("cm^2/V/s")
    Cr=2.23E17*U("cm^(-3)")
    Cs=6.10E20*U("cm^(-3)")
    pc=9.23E16*U("cm^(-3)")
    alpha=0.719*U("dimensionless")
    beta=2*U("dimensionless")
    
    #computing the mobility of electrons
    if NA!=0:
        mu_h=mu_1*np.exp(-pc/NA)+mu_max/(1+(NA/Cr)**alpha)-mu_2*((NA/Cs)**beta)/((NA/Cs)**beta+1) #we modified the last part of the equation to allow to compute when NA=0
    else: #in the case where NA=0 we need to remove the exponential
        mu_h=mu_max
        
    return mu_h


def carrier_concentrations(U,ND,NA,T,T_inf,kb,Eg): 
    '''This function computes the carrier concentrations for the donor and 
    acceptor contribution.
    
    This following is a combination of the model in Basu 2010 and Fu 2005.
    '''
    
    #computing the effective density of states in the valence and conduction bands (Nc and Nv) (this is from Fu 2005)
    Nc0=2.86E19*U("cm^(-3)")
    Nv0=2.66E19*U("cm^(-3)")
    Nc=Nc0*(T/T_inf)**1.5
    Nv=Nv0*(T/T_inf)**1.5
    
    #computing the thermally excited fee electrons and holes
    Nth=np.sqrt(Nc*Nv*math.exp(-Eg/(kb*T)))


    if ND!=0: #if n-type
        #parameters from Basu 2010
        A=0.0824*(T/T_inf)**(-1.622)
        N0=1.6E18*U("cm^(-3)")*(T/T_inf)**(0.7267)
        if ND<N0:
            B=0.4722*(T/T_inf)**(0.0652)
        else:
            B=1.23-0.3162*(T/T_inf)
            
        #computing the degree of ionization for electrons
        zeta_e=1-A*np.exp(-(B*(np.log(ND/N0)))**2)
        
        #computing the number of ionized atomes of the majority carrier
        n=zeta_e*ND
        
        #computing the majority carrier concentration
        Ne=0.5*(n+np.sqrt(n**2+4*Nth**2))
        
        #computing the minority carrier concentration
        Nh=Nth**2/Ne
    
    else: #if p-type
        #parameters from Basu 2010
        A=0.2364*(T/T_inf)**(-1.474)
        N0=1.577E18*U("cm^(-3)")*(T/T_inf)**(0.46)
        if NA<N0:
            B=0.433*(T/T_inf)**(0.2213)
        else:
            B=1.268-0.338*(T/T_inf)
            
        #computing the degree of ionization for holes
        zeta_h=1-A*np.exp(-(B*(np.log(NA/N0)))**2)

        #computing the number of ionized atoms of the majority carrier
        p=zeta_h*NA
        
        #computing the majority carrier concentration
        Nh=0.5*(p+np.sqrt(p**2+4*Nth**2))
        
        #computing the minority carrier concentration
        Ne=Nth**2/Nh
    
    
    return Ne,Nh

def F1(U,x): 
    '''This function evaluates the absorption coefficient associated with phonons
    of 212K equivalent energy (from Timans 1993).
    '''
    
    #computing factors that will act as the equivalent of an if condition (this removes the need for a loop making the code much faster)
    factor_1=x.m_as("eV")>0
    factor_2=x.m_as("eV")>0.0055
    
    #computing the function
    F1_x=factor_1*0.504*U("eV^(1/2)*cm^(-1)")*np.sqrt(x,dtype=complex)+factor_2*392*U("eV^(-1)*cm^(-1)")*(x-0.0055*U("eV"))**2
        
    return F1_x #in eV*cm^-1
    

def F2(U,x): 
    '''This function evaluates the absorption coefficient associated with phonons 
    of 670K equivalent energy (from Timans 1993).
    '''

    #computing factors that will act as the equivalent of an if condition (this removes the need for a loop making the code much faster)
    factor_1=x.m_as("eV")>0
    factor_2=x.m_as("eV")>0.0055
    
    #computing the function
    F2_x=factor_1*18.08*U("eV^(1/2)*cm^(-1)")*np.sqrt(x,dtype=complex)+factor_2*5760*U("eV^(-1)*cm^(-1)")*(x-0.0055*U("eV"))**2
        
    return F2_x #in eV*cm^-1


def alpha_3_calc(U,kb,E,Eg,T): 
    '''This function computes the absorption coefficient associated with phonons 
    of 1050K equivalent energy (from Timans 1993).
    '''

    #phonons temperature
    theta_3=1050*U("K")    
    
    #computing a factor that will act as the equivalent of an if condition (this removes the need for a loop making the code much faster)
    factor=E.m_as("eV")>=(Eg-kb*theta_3).m_as("eV")
    
    #computing the function
    alpha_3=factor*536*U("eV^(-1)*cm^(-1)")*(E-Eg+kb*theta_3)**2/(E*(np.exp(theta_3/T)-1))
    
    return alpha_3 #in cm^-1


def alpha_4_calc(U,kb,E,Eg,T): 
    '''This function computes the absorption coefficient associated with phonons 
    of 1420K equivalent energy (from Timans 1993).
    '''

    #phonons temperature
    theta_4=1420*U("K")    
    
    #computing a factor that will act as the equivalent of an if condition (this removes the need for a loop making the code much faster)
    factor=E.m_as("eV")>=(Eg-kb*theta_4).m_as("eV")
    
    #computing the function
    alpha_4=factor*988*U("eV^(-1)*cm^(-1)")*(E-Eg+kb*theta_4)**2/(E*(np.exp(theta_4/T)-1))
    
    return alpha_4 #in cm^-1


def alpha_IB_calc(U,lambd,Eg,hbar,kb,T): 
    '''This function computes the interband absorption using the model in Timans 1993
    '''
    
    #converting wavelength to energy
    E=2*hbar*U.pi*U.c/lambd
    
    #phonons energy equivalent temperatures
    theta_1=212*U("K")
    theta_2=670*U("K")
    
    #Some coefficients for the calculation
    A=E-Eg+kb*theta_1
    B=E-Eg-kb*theta_1
    C=E-Eg+kb*theta_2
    D=E-Eg-kb*theta_2
    
    #computing components of the alpha_IB function
    Coeff1=F1(U,A)/(np.exp(theta_1/T)-1)
    Coeff2=F1(U,B)/(1-np.exp(-theta_1/T))
    Coeff3=F2(U,C)/(np.exp(theta_2/T)-1)
    Coeff4=F2(U,D)/(1-np.exp(-theta_2/T))
    
    #components of the absorption coefficient which are associated with phonon absorption processes with characteristic energies equivalent to 1050K and 1420K
    alpha_3=alpha_3_calc(U,kb,E,Eg,T)
    alpha_4=alpha_4_calc(U,kb,E,Eg,T)
    
    #computing the total alpha_IB
    alpha_IB=1/E*(Coeff1+Coeff2+Coeff3+Coeff4)+alpha_3+alpha_4
    
    return alpha_IB #in cm^-1


def eps_r_calc(U,T): 
    '''This function computes the eps_r coefficient from Lee 2005.
    '''

    #some coefficients
    A=11.4447*U("dimensionless")
    B=2.7739E-4*U("K^-1")
    C=1.7050E-6*U("K^-2")
    D=-8.1347E-10*U("K^-3")
    
    #computing eps_r
    eps_r=A+B*T+C*T**2+D*T**3
    
    return eps_r


def g_calc(U,T): 
    '''This function computes the g coefficient from Lee 2005
    '''

    #some coefficients
    A=0.8948E-12*U("m^2")
    B=4.3977E-16*U("m^2*K^-1")
    C=7.3835E-20*U("m^2*K^-2")
    
    #computing eps_r
    g=A+B*T+C*T**2
    
    return g


def eta_calc(U,T): 
    '''This function computes the eta coefficient from Lee 2005
    '''

    #some coefficients
    A=-0.071*U("dimensionless")
    B=1.887E-6*U("K^-1")
    C=1.934E-9*U("K^-2")
    D=-4.544E-13*U("K^-3")
    
    #computing eps_r
    eta=np.exp(-3.0*(A+B*T+C*T**2+D*T**3))
    
    return eta


def eps_bl_calc(U,lambd,T,Eg,hbar,kb): 
    '''This function computes the epsilon_bl component from Lee 2005 and 
    the interband absorption using the model from Timans 1993
    '''

    #computing the interband absorption 
    alpha_IB=alpha_IB_calc(U,lambd,Eg,hbar,kb,T)
    
    #computing the extinction coefficient from the absorption coefficient
    kIB=alpha_IB*lambd/(4.0*U.pi)
    
    #computing some coefficient for the eps_bl calculation (from Lee 2005)
    eps_r=eps_r_calc(U,T)
    g=g_calc(U,T)
    eta=eta_calc(U,T)
    
    #computing the eps_bl coeff (here we are not sure it is ok to simply add the kIB component would have to verify this if we want to use this model for an absorber)
    eps_bl=(np.sqrt(eps_r+(g*eta)/lambd**2)+1j*kIB)**2
    
    return eps_bl
    

def epsilon_Si(lambd, U, T, ND, NA):
    '''This function computes the permittivity of Doped Silicon
    '''
    
    #constants
    el=U.elementary_charge#C
    m0=U.electron_mass #kg
    me=0.27*m0
    mh=0.37*m0
    epsilon0=U.vacuum_permittivity
    kb=U.k
    hbar=U.hbar
    c=U.c
    
    #frequency
    w=2*U.pi*c/lambd
    
    #reference temperature
    T_inf=300*U("K")
    
    #computing the bandgap energy (this is from Alex 1996 "Temperature dependence of the indirect energy gap in crystalline silicon")
    T_2=655*U("K")
    Eg=1.1692*U("eV")-0.00049*U("eV/K")*T**2/(T+T_2)
    
    #computing the epsilon_bl component
    eps_bl =eps_bl_calc(U,lambd,T,Eg,hbar,kb)
    
    #computing the scattering time of electron and holes
    tau_e,tau_h=tau_calc(U,el,me,mh,T,T_inf,ND,NA)
    
    #computing the carrier concentrations for the donnor and acceptor concentration
    Ne,Nh=carrier_concentrations(U,ND,NA,T,T_inf,kb,Eg)
    
    #computing the total permittivity
    eps_Si=eps_bl-(Ne*el**2/(epsilon0*me))/(w**2+1j*w/tau_e)-(Nh*el**2/(epsilon0*mh))/(w**2+1j*w/tau_h)
    
    return eps_Si