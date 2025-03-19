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

#importing the pint registry and seting up pint in matplotlib
U = pint.UnitRegistry()
U.setup_matplotlib(True)

#########User Variables####################
#temperature of the material
T=300*U("K")

####For Si###
#this is the concentration of donor and acceptor atoms in cm-3
#ND is the n-type semiconductor and NA is the p-type semiconductor
NA_Si=5e19*U("cm^(-3)")
ND_Si=0*U("cm^(-3)")



########################For InAs################################
#doping level of the layer
Nd_InAs=2.013e18*U("cm^(-3)")
#the entry is true if the layer is n-doped. If false, it is p-doped with N_A = Nd_InAs
N_doped=True
#Flag for the user to choose between the Kramers-Kronig relations and the approximation
model_choice="Approximation"#choose between "Kramers-Kronig" or "Approximation"
#Flag for the user to choose between computing the carrier concentration or 
#approximating it to the doping level
N_calc_choice=False#choose between True or False





def Plotting(w,epsilon,title): 
    '''
    Plot the real and imaginary parts of dielectric function epsilon at energies hbar*w.
    w is an array of frequencies
    title is a string used for the title of the plot.
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
    plt.plot((U.hbar*w).to('eV'), epsilon.real.to('dimensionless'), color="blue") 
    # Printing a title to the graph and the axes
    plt.xlabel("Photon energy [eV]")
    plt.ylabel("$\epsilon^\prime(\omega)$")
    plt.xscale('log')
    #making the ticks go inside the box and appear on all axis
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    
    ############################## Second plot ####################################
    plt.subplot(1, 2, 2)
    plt.loglog((U.hbar*w).to('eV'), epsilon.imag.to('dimensionless'), color="blue")
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




#closing all the existing plots
plt.close('all')





##############################frequency array generation########################
#energy range for the computation (eV)
Emin=0.01
Emax=6
E_vec=np.logspace(np.log10(Emin),np.log10(Emax),1600)*U("eV")

#converting to rad/s
w=E_vec/U.hbar


#####################################################################################################
##################################Calculating the permittivity#######################################
#####################################################################################################
epsilon_Si = permittivity_Si.epsilon_Si(w, U, T, ND_Si, NA_Si)
epsilon_InAs = permittivity_InAs.epsilon_InAs(w,Nd_InAs,T,N_doped,model_choice,N_calc_choice,U)

    

############################Plotting the results##############################
Plotting(w,epsilon_Si, "Si") #Si
Plotting(w,epsilon_InAs, "InAs") #InAs
