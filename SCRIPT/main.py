# # # # # # # # # # # # # # # # # # # # # # # #
#                                             # 
#       EVALUATION OF THERMAL STRESSES        #
#          IN ASPHALT CONCRETE SLABS          #
#            By: Benjamin J. Mailhé           #
#                 15-03-2020                  #
#              Version Aplha-0.1              #  
#                                             #
# # # # # # # # # # # # # # # # # # # # # # # #

###################
#   CLEAR SHELL   #
###################
import os
os.system('clear')
print('##################################')
print('# AC THERMAL STRESS CALC ROUTINE #')
print('##################################')

######################################
#   IMPORTATION OF USEFUL PACKAGES   #
######################################
import math
import cmath
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

################################
#   MATPLOTLIB CONFIGURATION   #
################################
#%matplotlib qt
#%matplotlib inline

###########################
#   LINK TO SUBROUTINES   #
###########################
import sys
sys.path.insert(1,'AC_THERMAL_STRESS/BIN/')

############################
#   FUNCTION IMPORTATION   #
############################
from input import paramin, indexin
from temp_profile import temp_profile_calculation as TPC
from temp_profile import time_discr,depth_discr
from plot_tempprofile import tempVStime as TvsTime
from plot_tempprofile import depthVStemp as ZvsT
from import_material import input_material

freqE=str(input('\nDo you want to treat E* in frequency? (y/N)\n'))

if freqE=='y' or freqE=='Y':
    from plot_material_Efreq import plot_input
    from optim_rawmat_Efreq import optim_loglog,optim_plot
    from tref_shift_Efreq import tref_sel,mean_shift,equiv_slope
    from optim_prony_Efreq import reduced_t_order,optim_prony,prony_plot
else:
    from plot_material import plot_input
    from optim_rawmat import optim_loglog,optim_plot
    from tref_shift import tref_sel,mean_shift,equiv_slope
    from optim_prony import reduced_t_order,optim_prony,prony_plot

from optim_wlf import optim_wlf,wlf_plot
from optim_arrh import optim_arrhenius,arrhenius_plot
from interconvertion import interconv_CCMC_Edyn,optim_interconv,interconv_plot
from linear_interp import timeinterp,tempinterp,plotinterp
from stress_calculation import tsecWLF,tsecArrh,stressprecalc
from python_stress import stresscalc, stresscalcFD
from plot_stress import stressplot

##############################################
#   INPUTS FOR THERMAL PROFILE CALCULATION   #
##############################################
while True:
        
    default=str(input('\nDo you want to use example material properties? (Y/n)\n'))
            
    if default=='n' or default=='N':

        kac=paramin('\nThermal conductivity Asphalt Concrete Mixture [W/(m.K)] = ')
        aac=paramin('Thermal diffusivity Asphalt Concrete Mixture  [m^2/s] = ')
        kgb=paramin('Thermal conductivity Granular Base [W/(m.K)] = ')
        agb=paramin('Thermal diffusivity Granular Base [m^2/s] = ')

        z_cm=paramin('\nAsphalt Concrete layer thickness [cm] = ')

        tmax=paramin('\nMax temperature [ºC]= ')
        tmin=paramin('Min temperature [ºC]= ')
        tmean=paramin('Mean temperature [ºC]= ')
        
    else:

        # ACmix prop default values
        #kac=1.1       
        kac=0.0021    # Qingwen
        #aac=4.69e-7   
        aac=1.38      # Qingwen
        # GB prop default values
        #kgb=1.4       
        kgb=0.003     # Qingwen
        #agb=3.5e-7    
        agb=1         # Qingwen
        # AC layer thickness example value
        z_cm=12.5
        # Temperature example values
        tmax=21
        tmin=10
        tmean=16

        print('\nThermal conductivity Asphalt Concrete Mixture [W/(m.K)] =',kac)
        print('Thermal diffusivity Asphalt Concrete Mixture [m^2/s] =',kgb)
        print('Thermal conductivity Granular Base [W/(m.K)] =',aac)
        print('Thermal diffusivity Granular Base [m^2/s] =',agb)
        
        print('\nAsphalt Concrete layer thickness [cm] =',z_cm)
        
        print('\nMax temperature [ºC] =',tmax)
        print('Min temperature [ºC] =',tmin)
        print('Mean temperature [ºC] =',tmean)
        
    correct=str(input('\nIs that correct? (y/N)\n'))
    if correct=='y' or correct=='Y': break

###########################################
#   INPUTS FOR STRESS MODEL CALCULATION   #
###########################################
while True:

    default=str(input('\nDo you want to use example calculation parameters? (Y/n)\n'))

    if default=='n' or default=='N':
        tday=indexin('\nNumber of days to compute [days] = ')
        tstep=paramin('Time step of the model [hours] = ')
        zdiscr=paramin('Depth steps of the model [cm] = ')

    else:
        tday=1 # number of days to compute
        tstep=2 # time step (in hour)
        zdiscr=2.5 # depth discretization in [cm]

        print('\nNumber of days to compute [days] = ',tday)
        print('Time step of the model [hours] = ',tstep)
        print('Depth steps of the model [cm] = ',zdiscr)

    # We should always compute 2 days and eliminate the first day
    # so we avoid the discontinuity.
    tday+=1

    correct=str(input('\nIs that correct? (y/N)\n'))
    if correct=='y' or correct=='Y': break

###############################################
#   INPUTS FOR STRESS EVOLUTION CALCULATION   #
###############################################
while True:

    default=str(input('\nDo you want to use example mixture properties? (Y/n)\n'))

    if default=='n' or default=='N':
        bmixin=str(input('\nDo yo already have Bmix ? (Y/n)\n'))

        if bmixin=='n' or bmixin=='N':
            Vin=str(input('\nDo yo already have VMA and Veff ? (Y/n)\n'))

            if Vin=='n' or Vin=='N':
                lbin=paramin('\nlinear coeff thermal contraction Binder [1/K] = ')
                lag=paramin('linear coeff thermal contraction Agg [1/K] = ')
                gsb=paramin('\nGsb aggregate bulk specific gravity = ')
                gb=paramin('Gb binder specific gravity = ')
                gmm=paramin('Gmm asphalt mixture maximum specific gravity =')
                av=paramin('\nAir volume [%] = ')
                bc=paramin('Binder content [%] = ')

                gmb=(1-av/100)*gmm
                vma=round(100-gmb*(100-bc)/gsb,2)
                gse=(100-bc)/(100/gmm-bc/gb)
                veff=round(gmb*(100-bc)/gse,2)
                #veff=round(gmb*(100/gmm-bc/gb),2)
                bmix=(vma*lbin+veff*lag)/300

            else:
                lbin=paramin('\nlinear coeff thermal contraction Binder [1/K] = ')
                lag=paramin('linear coeff thermal contraction Agg [1/K] = ')
                vma=paramin('\nVMA  intergranular void content in aggregate [%] = ')
                veff=paramin('Veff aggregate volume in mix [%] = ')

            bmix=(vma*lbin+veff*lag)/300
            print('\nBmix linear coefficient of thermal contraction of asphalt mix [1/C] = '+f'{bmix:.3e}')

        else:
            bmix=paramin('Bmix linear coefficient of thermal contraction of asphalt mix [1/C] = ')

    else:
        # Bmix example value
        bmix=2.351e-05   
        print('\nBmix linear coefficient of thermal contraction of asphalt mix [1/C] = '+f'{bmix:.3e}')
    
    correct=str(input('\nIs that correct? (y/N)\n'))
    if correct=='y' or correct=='Y': break

#######################################
#   TEMPERATURE PROFILE CALCULATION   #
#######################################
vecTime=time_discr(tday,tstep)
vecDepth=depth_discr(z_cm,zdiscr)
temp_profile=TPC(vecTime,vecDepth,kac,kgb,aac,agb,tmax,tmin,tmean)

#######################################################
#   TEMPERATURE PROFILE VISUALIZATION & EXPORTATION   #
#######################################################
TvsTime(temp_profile,vecTime,vecDepth*1e2)

#########################################
#   DEPTH vs. TEMPERATURE (NO EXPORT)   #
#########################################
ZvsT(temp_profile,tstep,vecDepth*1e2)

########################################
#   INPUT MATERIAL DATA - CREEP / E*   #
########################################
material_data=input_material()
temperature_data=material_data.iloc[:,0].unique()
if material_data.columns[2]=='compliance_gpa^(-1)':
    typeI=1 # Creep Compliance input
else:
    typeI=0 # E* input

plot_input(material_data,typeI,temperature_data)
print('\n',material_data)

############################################
#   INPUT MATERIAL MANIP AND CURVE OPTIM   #
############################################
raw_coeff,raw_stderr=optim_loglog(material_data,typeI,temperature_data)
optim_plot(material_data,typeI,temperature_data,raw_coeff,raw_stderr)

######################
#   TREF SELECTION   #
######################
temp_minus_tref,tref_index,tref=tref_sel(temperature_data)

#####################################
#   MEAN SHIFT-FACTOR CALCULATION   #
#####################################
shifting_coeff=mean_shift(material_data,typeI,temp_minus_tref,tref_index,raw_coeff)

######################################
#   SHIFT USING EQUIV-SLOPE METHOD   #
######################################
reduced_time=equiv_slope(material_data,typeI,temp_minus_tref,tref,shifting_coeff)

##############################
#   WLF MODEL OPTIMIZATION   #
##############################
coeff_wlf,stderr_wlf=optim_wlf(shifting_coeff,temperature_data,tref,tref_index)

####################################
#   ARRHENIUS MODEL OPTIMIZATION   #
####################################
coeff_arrh,stderr_arrh=optim_arrhenius(shifting_coeff,temperature_data,tref,tref_index)

while True:
    logat_sel=0
    wlf_plot(temperature_data,shifting_coeff,tref,tref_index,coeff_wlf,stderr_wlf)
    arrhenius_plot(temperature_data,shifting_coeff,tref,tref_index,coeff_arrh,stderr_arrh)

    while logat_sel!=1 and logat_sel!=2:
        logat_sel=int(input('\nWhich log(at) model do you want to use?\n1 - WLF\n2 - Arrhenius\nSelection = '))

    correct=str(input('\nAre you sure ? (y/N)\n'))
    if correct=='y' or correct=='Y': break

############################################
#   PRONY-J/E* SERIE OPTIMIZATION & PLOT   #
############################################
tr_order,MC_order=reduced_t_order(reduced_time,material_data)
coeff_prony,stderr_prony,branch=optim_prony(tr_order,MC_order,typeI)
prony_plot(tr_order,MC_order,typeI,coeff_prony,stderr_prony,branch)

############# IF CC-MASTER-CURVE #############
if typeI==1:

    ############################
    #   INTERCONVERTION > E*   #
    ############################
    Edyn_IC=interconv_CCMC_Edyn(tr_order,MC_order,coeff_prony,branch)

    ##################################
    #   PRONY-E SERIE OPTIMIZATION   #
    ##################################
    coeff_ICprony,stderr_ICprony,branch_ICprony=optim_interconv(tr_order,Edyn_IC)
    interconv_plot(tr_order,Edyn_IC,coeff_ICprony,stderr_ICprony,branch_ICprony)

##############################################

############################
#   LINEAR INTERPOLATION   #
############################
vecTimeInterp=timeinterp(vecTime)
vecTempInterp=tempinterp(vecTime,vecTimeInterp,temp_profile,vecDepth)
# plotinterp(vecTime,temp_profile,vecTimeInterp,vecTempInterp)

# SEEMS TO BE A BIT USELESS ?
# GIVES ERRONEOUS RESULTS IN FD IMPLEMENTATION
# CHECK INFLUENCE ON PSEUDO-VAR CALCULATION AT Tref =/= -20

#vecTimeInterp=vecTime
#vecTempInterp=temp_profile

##################################
#   THERMAL STRESS CALCULATION   #
##################################
if logat_sel==1:
    logat=tsecWLF(vecTempInterp,coeff_wlf,tref)
else:
    logat=tsecArrh(vecTempInterp,coeff_arrh,tref)

RedTime,Delta_RedTime,Etot,Delta_Etot=stressprecalc(vecTimeInterp,vecTempInterp,vecDepth,logat,bmix,tmean)

Eref=1

if typeI==1:
    EpronyBranch=branch_ICprony
    EpronyCoeff=coeff_ICprony
else:
    EpronyBranch=branch
    EpronyCoeff=coeff_prony

CalculatedStressPSEUDO=stresscalc(vecTimeInterp,vecDepth, \
                                  Delta_RedTime,Etot,Delta_Etot, \
                                  EpronyCoeff,EpronyBranch,Eref)

CalculatedStressFD=stresscalcFD(vecTimeInterp,vecDepth, \
                                Delta_RedTime,Etot,Delta_Etot, \
                                EpronyCoeff,EpronyBranch)

stressplot(vecTimeInterp,vecDepth*1e2,CalculatedStressPSEUDO)

stressplot(vecTimeInterp,vecDepth*1e2,CalculatedStressFD)