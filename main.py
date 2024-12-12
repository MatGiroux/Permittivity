# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:57:08 2021

@author: girm0
"""
import permittivity_Si
import permittivity_InAs
import numpy as np
import matplotlib.pyplot as plt
import pint


def Plotting(hbar,w,epsilon,title): 
    '''This function generates a plot of the real and imaginary part of the permittivity
    '''

    '########################################################### Plot data #######################################################################'

    """
    Global  settings for all plots in paper
    """
    
    font = {'family':'Arial','weight':'normal','size':'10'}
    plt.rc('font', **font)
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['text.usetex'] = False
    plt.rcParams['legend.borderpad'] =0.0
        
    ##################################################################################################################
    ################################################Plotting the results##############################################
    ##################################################################################################################
    
    fig = plt.figure(figsize=(6.6, 3.0),dpi=300) 
    
    
    ############################### first plot ####################################
    plt.subplot(1, 2, 1)
    plt.plot((hbar*w).to('eV'), epsilon.real.to('dimensionless'), color="blue") 
    # Printing a title to the graph and the axes
    plt.xlabel("Photon energy [eV]")
    plt.ylabel("$\epsilon^\prime(\omega)$")
    plt.xscale('log')
    #making the ticks go inside the box and appear on all axis
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    
    ############################## Second plot ####################################
    plt.subplot(1, 2, 2)
    plt.loglog((hbar*w).to('eV'), epsilon.imag.to('dimensionless'), color="blue")
    # Printing a title to the graph and the axes
    plt.xlabel("Photon energy [eV]")
    plt.ylabel("$\epsilon^{\prime\prime}(\omega)$")
    #making the ticks go inside the box and appear on all axis
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    
    #generating title
    plt.suptitle(title)
    
    
    plt.tight_layout(pad=0.2)
    plt.show()
    
    return



#importing the pint registry and seting up pint in matplotlib
U = pint.UnitRegistry()
U.setup_matplotlib(True)


#closing all the existing plots
plt.close('all')


########################################User Variables######################################
#temperature of the material
T=300*U("K")

###########################For Si################################
#this is the concentration of donor and acceptor atoms in cm-3
#ND is the n-type semiconductor and NA is the p-type semiconductor
NA=5e19*U("cm^(-3)")
ND=0*U("cm^(-3)")



########################For InAs################################
#doping level of the layer
N=2.013e18*U("cm^(-3)")
#the entry is true if the layer is n-doped
N_doped=True
#Flag for the user to choose between the Kramers-Kronig relations and the approximation
model_choice="Approximation"#choose between "Kramers-Kronig" or "Approximation"





#some constants
c=U.c
hbar=U.hbar

##############################frequency array generation########################
#energy range for the computation
Emin=0.01
Emax=6
E_vec=np.logspace(np.log10(Emin),np.log10(Emax),1600)*U("eV")

#converting to rad/s
w=E_vec/hbar

#converting frequency to wavelength
lambd=2*U.pi*c/w


#####################################################################################################
##################################Calulating the permittivity########################################
#####################################################################################################
epsilon_Si = permittivity_Si.epsilon_Si(lambd, U, T, ND, NA)
epsilon_InAs = permittivity_InAs.epsilon_InAs(lambd,N,T,N_doped,model_choice,U)

    

############################Plotting the results##############################
Plotting(hbar,w,epsilon_Si, "Si") #Si
Plotting(hbar,w,epsilon_InAs, "InAs") #InAs
