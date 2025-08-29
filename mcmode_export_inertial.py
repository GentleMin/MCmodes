# -*- coding: utf-8 -*-


import os, h5py
import numpy as np
import pandas as pd
import scipy.special as specfun

import models
from operators.worland_transform import WorlandTransform
from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM
from fields import VectorFieldSingleM

dir_name = './runs/Axial_Le1e-2_Eeta1e-2_m3/'

nr, maxnl, m_val = 93, 93, 3

model = models.MagnetoCoriolis_Spin(nr, maxnl, m_val, inviscid=True,
    induction_eq_params={'galerkin': False, 'ideal': False, 'boundary_condition': True})


# eigenvals = list()
# eigenvecs = list()
with h5py.File(os.path.join(dir_name, "eigenmodes.h5"), 'r') as fread:
    # n_gps = len(fread.keys())
    # for i_gps in range(n_gps):
    eigenvals = fread[f"res_0"]["eigenvals"][()]
    eigenvecs = fread[f"res_0"]["eigenvecs"][()]


# rg = np.linspace(0, 1, 201)
rg = specfun.roots_chebyt(401)[0]
rg = rg[rg.size//2:]
tg = np.linspace(0, np.pi, 201)
rr, tt = np.meshgrid(rg, tg)

worland_transform = WorlandTransform(nr, maxnl, m_val, r_grid=rg)
legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m_val, tg)

mode_idx = 0
eigenval_collect = []
u_collect = []
b_collect = []

for mode_idx in range(eigenvals.size):
# for mode_idx in range(3):

    usp = VectorFieldSingleM(nr, maxnl, m_val, eigenvecs[:model.dim['u'], mode_idx])
    bsp = VectorFieldSingleM(nr, maxnl, m_val, eigenvecs[model.dim['u']:, mode_idx])
    norm = np.sqrt(usp.energy)
    usp.normalise(norm)
    bsp.normalise(norm)
    
    uphy_md = usp.physical_field(worland_transform, legendre_transform)
    bphy_md = bsp.physical_field(worland_transform, legendre_transform)
    u_sph = uphy_md.data
    b_sph = bphy_md.data
    
    eigenval_collect.append(eigenvals[mode_idx])
    u_collect.append(np.stack([u_sph['r'], u_sph['theta'], u_sph['phi']], axis=0))
    b_collect.append(np.stack([b_sph['r'], b_sph['theta'], b_sph['phi']], axis=0))
    print('{}/{} finished.'.format(mode_idx+1, eigenvals.size))


with h5py.File(os.path.join(dir_name, 'eigen_export_filter.h5'), 'x') as fh5:
    fh5.create_dataset('Eigenvalues', data=np.asarray(eigenval_collect))
    fh5.create_dataset('Meridion_slices_V', data=np.asarray(u_collect))
    fh5.create_dataset('Meridion_slices_B', data=np.asarray(b_collect))
    fh5.create_dataset('Coordinates', data=np.stack([rr, tt, np.zeros_like(rr)], axis=0))
