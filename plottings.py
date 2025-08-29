# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal, Optional


def plot_spectrum(datasets: List, ax: plt.Axes, loglog: bool = True, freq_abs: bool = True, 
    xlim: Optional[List] = None, ylim: Optional[List] = None, 
    xlabel: str = r"$|\omega| = |\mathrm{Im}[\lambda]|$", 
    ylabel: str = r"$\sigma = -\mathrm{Re}[\lambda]$"):
    """Plot spectrum in omega-sigma (frequency-growth/decay rate) space
    """
    for ds in datasets:
        if loglog or freq_abs:
            im = ax.scatter(np.abs(np.imag(ds["vals"])), -np.real(ds["vals"]), **ds["style"])
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, which="both")
    
    return im
