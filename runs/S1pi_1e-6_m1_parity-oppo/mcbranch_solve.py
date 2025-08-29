# -*- coding: utf-8 -*-

import numpy as np
import sys, os, h5py

from operators import polynomials as poly
import models, utils, tools

current_dir = os.path.dirname(__file__)

# Setting up
E_eta = 1e-6
Elsasser = 1.
m_val = 1
bg_modes = [poly.SphericalHarmonicMode("pol", 1, 0, "1/5 Sqrt[pi/3] r(5 - 3r^2)")]
mod_setup_MC = models.MagnetoCoriolis
parity = "none"
output_fname = os.path.join(current_dir, "eigenmodes")


tools.print_heading(f"Calculating eigenmodes for m = {m_val}, E_eta = {E_eta:.2e}, Elsasser = {Elsasser: .2e}", 
    prefix='\n', suffix='\n', lines="over", char='=')

resolution = (61, 61, m_val)
branch_targets = [
    -1.0e+1 - 2.0e-1j,
    -5.0e+1 + 6.0e-1j,
    -2.0e+1 - 2.0e+0j,
    -5.0e+1 + 2.0e+1j,
    -1.1e+2 + 4.0e+1j,
    -1.6e+2 + 7.0e+1j,
    -2.8e+2 + 1.8e+2j,
    -4.0e+2 + 3.7e+2j,
    -6.0e+2 + 6.0e+2j,
    -8.0e+2 + 9.0e+2j,
    -1.0e+3 + 1.5e+3j,
    -1.2e+3 + 1.9e+3j,
    -1.5e+3 + 2.5e+2j
]
k_nearest = 5

if parity == "none":
    eigenvals = list()
    eigenvecs = list()
else:
    eigenvals_QP = list()
    eigenvecs_QP = list()
    eigenvals_DP = list()
    eigenvecs_DP = list()

    
nr, maxnl, m = resolution
tools.print_heading(f"Resolution (L, N)=({maxnl}, {nr})", prefix='\n', suffix='\n', lines="over", char='-')

with utils.Timer("Build operator"):
    
    mod = mod_setup_MC(
        nr, maxnl, m, inviscid=True,
        induction_eq_params={'galerkin': False, 'ideal': False, 'boundary_condition': True}
    )
    if parity == "none":
        A, B = mod.setup_operator(
            field_modes=bg_modes, setup_eigen=True, magnetic_ekman=E_eta, elsasser=Elsasser
        )
    else:
        op_dp, op_qp = mod.setup_operator(
            field_modes=bg_modes, setup_eigen=True, magnetic_ekman=E_eta, elsasser=Elsasser, 
            parity=True, u_parity=parity
        )

if parity == "none":
    print(f"K shape: {A.shape[0]}x{A.shape[1]}\n"
        f"M shape: {B.shape[0]}x{B.shape[1]}")
    with utils.Timer("Solve eigensystem"):
        for target_val in branch_targets:
            tools.print_heading(f"Target = {target_val}", prefix='\t', lines="none")
            w, v = utils.single_eig(A, B, target=target_val, nev=k_nearest)
            eigenvals.append(w)
            eigenvecs.append(v)

else:
    print(f"DP system: {op_dp[0].shape[0]}x{op_dp[0].shape[1]}")
    with utils.Timer("Solve DP eigensystem"):
        for target_val in branch_targets:
            tools.print_heading(f"Target = {target_val}", prefix='\t', lines="none")
            w, v = utils.single_eig(op_dp[0], op_dp[1], target=target_val, nev=k_nearest)
            eigenvals_DP.append(w)
            eigenvecs_DP.append(v)
    
    print(f"QP system: {op_qp[0].shape[0]}x{op_qp[0].shape[1]}")
    with utils.Timer("Solve QP eigensystem"):
        for target_val in branch_targets:
            tools.print_heading(f"Target = {target_val}", prefix='\t', lines="none")
            w, v = utils.single_eig(op_qp[0], op_qp[1], target=target_val, nev=k_nearest)
            eigenvals_QP.append(w)
            eigenvecs_QP.append(v)


if output_fname is not None:

    os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    with h5py.File(output_fname + '.h5', 'a') as fwrite:
        n_gps = len(fwrite.keys())
        for i_target, target_val in enumerate(branch_targets):
            gp = fwrite.create_group(f"eigenmode_target{n_gps}")
            gp.attrs["L"] = resolution[0]
            gp.attrs["N"] = resolution[1]
            gp.attrs["m"] = resolution[2]
            gp.attrs["target"] = target_val
            if parity == "none":
                gp.create_dataset("eigenvals", data=eigenvals[i_target])
                gp.create_dataset("eigenvecs", data=eigenvecs[i_target])
            else:
                ds = gp.create_dataset("eigenvals_DP", data=eigenvals_DP[i_target])
                ds.attrs["u_parity"] = parity
                gp.create_dataset("eigenvecs_DP", data=eigenvecs_DP[i_target])
                ds = gp.create_dataset("eigenvals_QP", data=eigenvals_QP[i_target])
                ds.attrs["u_parity"] = parity
                gp.create_dataset("eigenvecs_QP", data=eigenvecs_QP[i_target])
            n_gps += 1

    tools.print_heading(f"Eigenvectors saved to {output_fname}.h5", 
        prefix='\n', suffix='\n', lines="under", char='=')



