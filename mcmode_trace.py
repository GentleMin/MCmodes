# -*- coding: utf-8 -*-
"""Trace eigenmode
"""


import os, h5py
import pandas as pd
import numpy as np

import models, tools, utils
from operators.polynomials import SphericalHarmonicMode


def load_seed(fname):
    df_targets = pd.read_table(fname, header=3, delimiter='\s+')
    eval_seeds = df_targets['Value'].to_numpy().astype(np.complex128)
    return eval_seeds


# System
sys_MC = models.MagnetoCoriolis_Alfven

# Physical params
Le = 1e-4
Lu = 2e+4
m_val = 3

# Background field
bg_modes = [SphericalHarmonicMode("pol", 2, 0, "1/4 Sqrt[3/26] r^2(5r^2 - 7)")]

# Resolutions
res_list = [
    (63, 63, m_val),
]

# io
# current_dir = os.path.dirname(__file__)
current_dir = './runs/QGP-SL2N2_Le1e-4_Lu2e+4_m3/'
seed_fname = os.path.join(current_dir, 'eigenspectra_describe.txt')
output_fname = os.path.join(current_dir, "eigenmodes_traced.h5")


if __name__ == '__main__':
    
    eig_seeds = load_seed(seed_fname)
    
    eig_vals = list()
    eig_vecs = list()
    eig_vals.append(eig_seeds)
    
    for i_res, res in enumerate(res_list):
        
        nr, maxnl, m = res
        eig_targets = eig_vals[-1]
        tools.print_heading(f"Resolution (L, N)=({maxnl}, {nr})", prefix='\n', suffix='\n', lines="over", char='-')
        
        mod_MC = sys_MC(res[0], res[1], res[2], inviscid=True, 
            induction_eq_params={'galerkin': False, 'ideal': False, 'boundary_condition': True})
        K, M = mod_MC.setup_operator(field_modes=bg_modes, setup_eigen=True, lehnert=Le, lundquist=Lu)
        
        eig_vals_tmp = np.zeros_like(eig_targets)
        eig_vecs_tmp = np.zeros((K.shape[0], eig_targets.shape[0]), dtype=np.complex128)
        
        for i_target, target in enumerate(eig_targets):
            w, v = utils.single_eig(K, M, target=target, nev=1)
            eig_vals_tmp[i_target] = w
            eig_vecs_tmp[:, i_target] = v[:, 0]
        
        print(eig_vals_tmp)
        
        eig_vals.append(eig_vals_tmp)
        eig_vecs.append(eig_vecs_tmp)
    
    with h5py.File(output_fname, 'x') as fwrite:
        n_gps = len(fwrite.keys())
        # n_gps = 0
        for i_res, res in enumerate(res_list):
            gp = fwrite.create_group(f"Resolution_{n_gps}")
            gp.attrs["N"] = res[0]
            gp.attrs["L"] = res[1]
            gp.attrs["m"] = res[2]
            gp.create_dataset("eigenvals", data=eig_vals[i_res])
            gp.create_dataset("eigenvecs", data=eig_vecs[i_res])
            n_gps += 1
        

