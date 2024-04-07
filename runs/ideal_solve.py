# -*- coding: utf-8 -*-

import numpy as np
import sys, os, h5py

from operators import polynomials as poly
import models, utils, tools


# Setting up
Le = 1e-4
m_val = 1
bg_modes = [poly.SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")]
mod_setup_MC = models.IdealMagnetoCoriolis
parity = "same"
# output_fname = "./out/T1pi__eigenspec__m{:d}__E_eta_{:.1e}__{:s}_parities".format(m_val, E_eta, parity)
output_fname = None


tools.print_heading(f"Calculating eigen spectrum for m = {m_val}, Le = {Le:.2e}", 
    prefix='\n', suffix='\n', lines="over", char='=')

resolutions = [
    (11, 11, m_val),
    (21, 21, m_val),    
    # (31, 31, m_val),
    # (41, 41, m_val)
]
if parity == "none":
    spectra = list()
else:
    spectra_QP = list()
    spectra_DP = list()

for i_res, res in enumerate(resolutions):
    
    nr, maxnl, m = res
    tools.print_heading(f"Resolution (L, N)=({maxnl}, {nr})", prefix='\n', suffix='\n', lines="over", char='-')
    
    with utils.Timer("Build operator"):
        
        mod = mod_setup_MC(
            nr, maxnl, m, inviscid=True,
            induction_eq_params={'galerkin': False, 'ideal': True, 'boundary_condition': False}
        )
        if parity == "none":
            A, B = mod.setup_operator(
                field_modes=bg_modes, setup_eigen=True, lehnert=Le
            )
        else:
            op_dp, op_qp = mod.setup_operator(
                field_modes=bg_modes, setup_eigen=True, lehnert=Le,
                parity=True, u_parity=parity
            )
    
    if parity == "none":
        print(f"K shape: {A.shape[0]}x{A.shape[1]}\n"
            f"M shape: {B.shape[0]}x{B.shape[1]}")  
        with utils.Timer("Solve eigensystem"):
            w = utils.full_spectrum(A, B)
        spectra.append(w)
        del A, B
    
    else:
        print(f"DP system: {op_dp[0].shape[0]}x{op_dp[0].shape[1]}")
        with utils.Timer("Solve DP eigensystem"):
            w = utils.full_spectrum(op_dp[0], op_dp[1])
        spectra_DP.append(w)
        
        print(f"QP system: {op_qp[0].shape[0]}x{op_qp[0].shape[1]}")
        with utils.Timer("Solve QP eigensystem"):
            w = utils.full_spectrum(op_qp[0], op_qp[1])
        spectra_QP.append(w)


if output_fname is not None:

    os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    with h5py.File(output_fname + '.h5', 'x') as fwrite:
        for i_res, res in enumerate(resolutions):
            gp = fwrite.create_group(f"eigenspec_res{i_res}")
            gp.attrs["L"] = res[0]
            gp.attrs["N"] = res[1]
            gp.attrs["m"] = res[2]
            if parity == "none":
                gp.create_dataset("spectrum", data=spectra[i_res])
            else:
                ds = gp.create_dataset("spectrum_DP", data=spectra_DP[i_res])
                ds.attrs["u_parity"] = "same"
                ds = gp.create_dataset("spectrum_QP", data=spectra_QP[i_res])
                ds.attrs["u_parity"] = "same"
