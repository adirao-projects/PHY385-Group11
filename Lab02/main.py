#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:28:07 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c


HENE_wl = 632.8e-9 # m 
HENE_fq = c/HENE_wl

def fringe_visibility(signal):
    numer = np.max(signal) - np.min(signal)
    denom = np.max(signal) + np.min(signal)
    print(numer)
    print(denom)
    return numer/denom
        

def I0(t, omega):
    return np.cos(omega*t)


def timedelay(d):
    return d/c


def Itot(omega, I0, tau):
    return 2*I0*(1+np.cos(omega*tau))


def I0d():
    d_vals = np.linspace(0, 0.10, 100)

    tau_vals = d_vals / c

    omega = HENE_fq
    
    intensity = Itot(omega, 1, tau_vals)
    
    print(intensity)
    
    plt.figure(figsize=(20,7))
    plt.title('Path Difference and Incident Intensity')
    plt.plot(d_vals, intensity)
    # plt.axvline(0.01, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(0.03, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(0.05, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(0.07, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(0.09, color='red', alpha=0.5, linestyle='dashed')
    plt.grid('on')
    plt.xlabel('Path Difference (m)')
    plt.ylabel('Intensity ($I_0$)')

def function():
    d_vals = np.linspace(10, 30, 5)
    omega = HENE_fq
    
    
    t_vals = np.linspace(0, 1, 100)
    
    tau_vals = timedelay(d_vals)
    
    init_intensity = I0(t_vals, omega)
    
    fring_vis = []
    
    for tau in tau_vals:
        intensity = Itot(omega, init_intensity, tau)
        plt.figure()
        #print(f'd = {tau*c}')
        plt.title(f'd = {tau*c}')
        plt.plot(t_vals, init_intensity)
        plt.plot(t_vals, intensity)
        plt.show()
        fv = fringe_visibility(intensity)
        print(fv)
        fring_vis.append(fv)
    
    plt.figure()
    plt.plot(tau_vals, fring_vis)
    
    
    
if __name__ == '__main__':
    print(HENE_fq)
    I0d()
