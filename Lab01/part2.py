#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:33:31 2025

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


def model_func_wp(theta, I0, phi, phase):
    return I0*((np.cos(phase*(theta+phi)))**2)

def model_func_qwp(theta, I0, phi):
    return 4*I0*((np.cos(theta+phi))**2)*((np.sin(theta+phi))**2)

def model_func_hwp(theta, I0, phi):
    return I0*((np.sin(2*theta+phi))**2)

def deg_rad(angle):
    return (angle/180)*(np.pi)
    
def rad_deg(angle):
    return (angle/np.pi)*(180)


def fit_plot(df, plate):
    
    xdata = deg_rad(df[plate].to_numpy())
    ydata = df['W'].to_numpy()
    y_unc = df['Wu'].to_numpy()
    x_unc = deg_rad(df['Tu'].to_numpy())
    
    #plt.errorbar(xdata, ydata, yerr=y_unc, xerr=x_unc, fmt='o')
    
    #data = tk.curve_fit_data(xdata, ydata, fit_type='custom',
    #                  model_function_custom=model_func_qwp, 
    #                  uncertainty=y_unc, uncertainty_x=x_unc,
    #                  res=True, chi=True, guess=(8, 0.5, 1.5))
    
    #print(xdata)
    if plate == 'QWP':
        plate_name = 'Quarter'
        data = tk.curve_fit_data(xdata, ydata, fit_type='custom',
                          model_function_custom=model_func_qwp, 
                          uncertainty=y_unc, uncertainty_x=x_unc,
                          res=True, chi=True,)
    
        meta = {'title' : f"Recorded Intensity for {plate_name} Waveplate\n",
                'xlabel' : r'Waveplate Angle ($\theta$) (rad)',
                'ylabel' : 'Recorded Intensity (Volts)',
                'chisq' : data['chisq'],
                'fit-label': r"$I' = I_0 \cos^2 (\theta+\phi)\sin^2(\theta+\phi)$",
                'data-label': "Raw data",
                'save-name' : f'{plate_name}',
                'loc' : 'upper right'}
    
        
    elif plate == 'HWP':
        plate_name = 'Half'
        data = tk.curve_fit_data(xdata, ydata, fit_type='custom',
                          model_function_custom=model_func_hwp, 
                          uncertainty=y_unc, uncertainty_x=x_unc,
                          res=True, chi=True, )
        
        meta = {'title' : f"Recorded Intensity for {plate_name} Waveplate\n",
            'xlabel' : r'Waveplate Angle ($\theta$) (rad)',
            'ylabel' : 'Recorded Intensity (Volts)',
            'chisq' : data['chisq'],
            'fit-label': r"$I' = I_0 \sin^2 (2\theta+\phi)$",
            'data-label': "Raw data",
            'save-name' : f'{plate_name}',
            'loc' : 'upper right'}
        
    max_I0 = np.max(data['ploty'])
    idx_maxI0 = np.where(data['ploty'] == max_I0)
    max_theta = data['plotx'][idx_maxI0]
    print(f'theta+phi max = {max_theta}')
    
    tk.quick_plot_residuals(xdata, ydata, data['plotx'], data['ploty'], 
                            data['residuals'], meta=meta,
                            uncertainty=y_unc, uncertainty_x=x_unc,
                            save=True)


    return data['chisq'], data['popt'], data['pstd']

if __name__ == '__main__':

    # Part (a)
    df = load_data('part2a.csv')
    params = fit_plot(df, 'QWP')
    
    printvals = [r'$\chi_{red}^2'+f' = {params[0]}$']
    for i,v in enumerate([r'I_0', r'\phi', ]):
        printvals.append(f'${v} = {params[1][i]} \pm {params[2][i]}$')
        
    tk.block_print(printvals, 'Fit Paramters for QWP')
        
    # Part (b)
    df = load_data('part2b.csv')
    params = fit_plot(df, 'HWP')
    
    printvals = [r'$\chi_{red}^2'+f' = {params[0]}$']
    for i,v in enumerate([r'I_0', r'\phi']):
        printvals.append(f'${v} = {params[1][i]} \pm {params[2][i]}$')
        
    tk.block_print(printvals, 'Fit Paramters for HWP')

    