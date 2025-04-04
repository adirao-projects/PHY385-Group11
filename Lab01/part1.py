#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:35:21 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""

import numpy as np
import pandas as pd
import toolkit as tk
import matplotlib.pyplot as plt

# uncert in voltage +- 0.01 V

def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()
    
    # Uncertainty in Output Voltage
    w_uncert = np.full(df.shape[0]+1, 0.5)
    
    # Uncertainty in Angle
    th_uncert = np.full(df.shape[0]+1, 0.01)
    
    df['Wu'] = pd.Series(w_uncert)
    df['Tu'] = pd.Series(th_uncert)
    
    return df


def model_func_malus(theta, I0, phi):
    return I0*((np.cos(theta+phi))**2)

def deg_rad(angle):
    return (angle/180)*(np.pi)
    
def rad_deg(angle):
    return (angle/np.pi)*(180)


def fit_plot(df):
    
    xdata = deg_rad(df['Pb'].to_numpy())
    ydata = df['W'].to_numpy()
    y_unc = df['Wu'].to_numpy()
    x_unc = deg_rad(df['Tu'].to_numpy())
    
    #plt.errorbar(xdata, ydata, yerr=y_unc, xerr=x_unc, fmt='o')
    
    #print(xdata)
    
    data = tk.curve_fit_data(xdata, ydata, fit_type='custom',
                      model_function_custom=model_func_malus, 
                      uncertainty=y_unc, uncertainty_x=x_unc,
                      res=True, chi=True, guess=(7, np.pi/4))
    
    
    meta = {'title' : "Malus' Law for Two Polarizers\n",
            'xlabel' : r'Second Polarizer Angle ($\theta$) (rad)',
            'ylabel' : 'Recorded Intensity (Volts)',
            'chisq' : data['chisq'],
            'fit-label': r"$I' = I_0 \cos^2 (\theta+\phi)$",
            'data-label': "Raw data",
            'save-name' : 'Malus',
            'loc' : 'upper right'}
    
    tk.quick_plot_residuals(xdata, ydata, data['plotx'], data['ploty'], 
                            data['residuals'], meta=meta,
                            uncertainty=y_unc, uncertainty_x=x_unc,
                            save=True)
    
    max_I0 = np.max(data['ploty'])
    idx_maxI0 = np.where(data['ploty'] == max_I0)
    max_theta = data['plotx'][idx_maxI0]
    print(f'theta+phi max = {max_theta}')

    return data['chisq'], data['popt'], data['pstd']

if __name__ == '__main__':
    df = load_data('part1.csv')

    params = fit_plot(df)
    
    printvals = [r'$\chi_{red}^2'+f' = {params[0]}$']
    for i,v in enumerate([r'I_0', r'\phi$']):
        printvals.append(f'${v} = {params[1][i]} \pm {params[2][i]}$')
        
    tk.block_print(printvals, 'Fit Paramters for Two Polarizer')