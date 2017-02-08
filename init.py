#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:58:55 2017

@author: py14sts && py14jl

Script made as part of NMR advanced lab for analysing T1 and T2 values received
from NMR measurements of 20:1 Li concentration poly(ethylene oxide) at 
different molecular weights and temperatures.

Plots function of correlation time (tau) according to derivation from the BPP
theory. Finds roots of equation, given T1 and T2 values, thereby determining 
values for correlation time.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
import itertools
import scipy.constants as consts

def ingestData(filename):
    """
    Read data from given filename and append into appropriate arrays
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    temps = data[:,0]
    mass = data[:,1]
    t1 = data[:,2]
    t2 = data[:,3]
    return temps, mass, t1, t2

def bppFunc(t1, t2, w0, tau):
    """
    f(tau) according to the BPP theory equations for 1/T1 and 1/T2.
    See 2017 lab book for derivation
    """
    numer = (16 * (w0**2) * (tau**2)) + (10)
    denom = (12 * (w0**4) * (tau**4)) + (37 * (w0**2) * (tau**2)) + (10)
    const = -t2/t1
    func = numer/denom + const
    return func
    
def bppPlotFunc(t1, t2, roots):
    """
    Plot BPP function and x axis
    """
    t1, t2 = t1/1000, t2/1000
#    print("Plotting...")
    plt.cla()
    # x-axis value range
    tau_values = np.linspace(-2E-8, 2E-8, 1000)
    # Resonance frequency of proton
    w0 = 2*consts.pi*20E6
    # Plot
    real_roots = []
    plt.plot(tau_values, bppFunc(t1, t2, w0, tau_values), label="BPP Theory", c='c')
    for i in roots:
        if(i.real):
            plt.scatter(i.real, 0, color='r', marker='x')
            real_roots.append(i.real)
    
    larger_root, smaller_root = max(real_roots), min(real_roots)
    x1label, x2label = 'x1 = ' + str(format(smaller_root, '0.3e')) + ' s', 'x2 = ' + str(format(larger_root, '0.3e')) + ' s'
    plt.annotate(x1label,
                 xy=(smaller_root, 0),
                 xytext=(0.05, 0.05),
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 horizontalalignment='left',
                 verticalalignment='bottom')
    plt.annotate(x2label,
                 xy=(larger_root, 0),
                 xytext=(0.95, 0.05),
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 horizontalalignment='right',
                 verticalalignment='bottom')
    plt.xlabel("Tau (s)")
    plt.ylabel("f(tau) from BPP therory.")
    plt.xlim(smaller_root * 2, larger_root * 2)
    plt.ylim(-0.1, 0.1)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend()
    plt.show()
    return
    
def findBppRoots(t1, t2):
    """
    Find the roots of the BPP function.
    Roots of equation derived using Symbolab equation solver.
    Gives 4 roots.
    """
    # variables renamed for convenience (a = w0)
    b = t2/t1
    a = 2*consts.pi*20E6
    comm_denom = 24 * (a**2) * b
    inner_root = cmath.sqrt((889 * (b**2)) - (704 * b) + 256)
    x1 = cmath.sqrt(-((37 * b) - 16 + inner_root)/comm_denom)
    x2 = (cmath.sqrt(-((37 * b) - 16 + inner_root)/comm_denom)) * -1.
    x3 = cmath.sqrt(-((37 * b) - 16 - inner_root)/comm_denom)
    x4 = (cmath.sqrt(-((37 * b) - 16 - inner_root)/comm_denom)) * -1.

    return [x1, x2, x3, x4]

def massRelation(t2, mass, temp):
    """
    Given t2 times, masses and temps, will plot a series of linear graphs
    between ln(mass) and 1/T2. Fits these linear graphs and returns dicts of 
    gradients and intercepts, indexed by their respectful temperatures.
    """
    plt.cla()
    marker = itertools.cycle(('*', '+', '^', 'o', 'D', 'x'))
    colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y'))
    masses, massI = np.unique(mass, return_index=True)
    temps, tempI = np.unique(temp, return_index=True)
    logMass = np.log(masses)
    xarray = np.linspace(min(logMass), max(logMass), 100)
    gradients = dict()
    intercepts = dict()
    for i in temps:
        ydata, xdata = [], []
        for j in range(len(temp)):
            if(i == temp[j]):
                ydata.append(t2[j]/1000)
                xdata.append(np.log(mass[j]))
        ydata = 1/np.asarray(ydata)
        thisColor = colors.next()
        plt.scatter(xdata, ydata, c=thisColor, marker=marker.next(), label=str(i)+'K')
        fit = np.polyfit(xdata, ydata, 1)
#        print("Gradient at T=" + str(i) + "K: " + str(fit[0]))
#        print("Intercept at T=" + str(i) + "K: " + str(fit[1]))
        gradients[i], intercepts[i] = fit[0], fit[1]
        fit_fn = np.poly1d(fit)
        plt.plot(xarray, fit_fn(xarray), c=thisColor)
        
    plt.xlabel("$ln(M_n)$")
    plt.ylabel("1/T2 ($s^-1$)")
    plt.title("Reciprocal T2 Time Against Natural Logarithm of Molecular Mass")
    plt.grid(True, which='both')
    plt.legend(loc=0)
    plt.show()
    return gradients, intercepts
    
def NaPlot(gradients, intercepts, monWeight=44.05):
    """
    Given dicts of gradients and intercepts (from massRelation function),
    calculates a value for Na for each temperature, then plots graph of Na
    against temperature. Fits a constant value to this (as expected from
    theory) and returns the constant value.
    
    """
    plt.cla()
    temps, gradientArr, icept = [], [], []
    for i in gradients:
        temps.append(i)
        gradientArr.append(gradients[i])
        
    for i in intercepts:
        icept.append(intercepts[i])
        
    logNam = np.asarray(icept)/np.asarray(gradientArr) * -1
    Nam = np.exp(logNam)
    Na = Nam/monWeight
    plt.scatter(temps, Na, label="Calculated $N_a$")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$N_a$")
    plt.title("$N_a$ Dependence on Temperature")
    plt.ylim(0, 3)
    fit = np.polyfit(temps, Na, 0)
    fit_fn = np.poly1d(fit)
    xarray = np.linspace(min(temps), max(temps), 100)
    plt.plot(xarray, fit_fn(xarray), label="Constant $N_a$ Fit")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()
    # Return constant value for Na
    return fit[0]
    
def findTau(t2, masses, temps, Na, Mn, m=44.05, d=1.76E-10, gamma=26.75E7):
    """
    Takes T2 times, masses, temps and a specific molecular mass. Calculates
    constant of proportionality according to [Macromolecules Vol. 31 p4951].
    Uses this constant of proportionality to calculate tau for at each temp
    for the given molecular mass. 
    Returns dict of tau values, indexed by respective temperatures.
    """
    # init storage
    tau_dict = dict()
    # Calculate delta
    delta_num = (gamma**2) * (consts.hbar) * (consts.mu_0)
    delta_denom = (8 * consts.pi) * (Na) * (d**3)
    delta = delta_num / delta_denom
    
    # Sub into proportionality equation (1/T2) = k(tau)
    uniqueTemps, tempI = np.unique(masses, return_index=True)
    k = (6 * (delta**2) * (np.log(Mn) - np.log(Na * m)))/(consts.pi)
    
    for i in range(len(masses)):
        if masses[i] == Mn:
            tauValue = 1000 / (t2[i] * k)
            tempValue = temps[i]
            tau_dict[tempValue] = tauValue
    
    return tau_dict
    
def plotTau(tau_dict):
    """
    Given dict of tau values and temps (from findTau function), plots tau vs
    temperature. Should fit to a polynomial. Gradient of the polynomial will 
    give value of tau_dot.
    """
    plt.cla()
    temps, taus = [], []
    for t in tau_dict:
        temps.append(t)
        taus.append(tau_dict[t])
        
    xarray = np.linspace(min(temps), max(temps), 100)
    plt.scatter(temps, taus, label="Calculated " + r"$\tau$")
    plt.grid(True, which='both')
    plt.xlabel("Temperature (K)")
    plt.ylabel(r"$\tau$" + " (s)")
    plt.title(r"$\tau$" + " Dependence on Temperature")
    plt.ylim(0, max(taus) * 1.1)
#    for i in range(1, len(taus)-1):
#        temp_taus = [taus[i-1], taus[i], taus[i+1]]
#        temp_temps = [temps[i-1], temps[i], temps[i+1]]
#        temp_fit = np.polyfit(temp_temps, temp_taus, 2)
#        temp_fit_fn = np.poly1d(temp_fit)
#        temp_xarray = np.linspace(temp_temps[0], temp_temps[2], 100)
#        plt.plot(temp_xarray, temp_fit_fn(temp_xarray))
        
    fit = np.polyfit(temps, taus, 4)
    fit_fn = np.poly1d(fit)
    plt.plot(xarray, fit_fn(xarray), label="Polynomial " + r"$\tau$" + " fit")
    plt.legend()
    plt.show()
    return(fit)
    
def tauDot(fit):
    """
    Polynomial fit of 3rd order - differentiate to get tau_dot
    Since arg is numpy fit, use np.polyder to differentiate.
    """
    diff_fn = np.polyder(fit)
    return diff_fn
    
    
if __name__ == "__main__":
    # Ingest data into arrays
    filename = str(raw_input("Enter filename of times: "))
    temps, mass, t1, t2 = ingestData(filename)
    # Get gradients and intercepts of 1/T2 vs ln(M) plots for different temps
    gradients, intercepts = massRelation(t2, mass, temps)
    bppPlotFunc(461., 447., findBppRoots(461., 447.))
    # Calculate Na and plot vs temps
    naConst = NaPlot(gradients, intercepts)
    print naConst
    # Calculate tau calculated Na and molecular mass 400
    tau_dict = findTau(t2, mass, temps, naConst, 300)
    tau_fit = plotTau(tau_dict)
    # Differentiate to get tau_dot
    tau_dot = tauDot(tau_fit)
    
    
    
    
    