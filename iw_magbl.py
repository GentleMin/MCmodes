# -*- coding: utf-8 -*-

import numpy as np
import sympy as sym
from sympy import diff, I, S
from scipy import special as specfun
import inertial_modes as iw
from inertial_modes import s, z, r, phi, theta
from op_sym import SphericalCoordinates, CylindricalCoordinates, cross, dot, conj
from typing import Literal

# from argparse import ArgumentParser

sph = SphericalCoordinates(r, theta, phi)
cyl = CylindricalCoordinates(s, phi, z)
map_cyl2sph = {s: r*sym.sin(theta), z: r*sym.cos(theta)}
alpha = sym.Symbol(r'\alpha')

def cyl2sph_vec(a, simplify=False):
    a_sph = [ai.subs(map_cyl2sph) for ai in a]
    a_sph = [
        a_sph[0]*sym.sin(theta) + a_sph[2]*sym.cos(theta), 
        a_sph[0]*sym.cos(theta) - a_sph[2]*sym.sin(theta),
        a_sph[1]
    ]
    if simplify:
        a_sph = [ai.simplify() for ai in a_sph]
    return a_sph


def iw_mode_sph(m, N, k, symm: Literal['S', 'A']):
    w0 = iw.eigenfreq_inviscid(N, m, parity=symm.lower(), n=18)[k]
    s0 = w0/2
    eig0 = I*w0
    u0_cyl = iw.eigenmode_poly_inviscid(N, parity=symm.lower())
    u0_cyl = [ui.subs({iw.m: m}) for ui in u0_cyl]
    u0_sph = cyl2sph_vec(u0_cyl)
    pars = {
        iw.elamb: eig0,
        iw.omega: w0,
        iw.sigma: s0,
        iw.m: m,
        alpha: (1 + I*sym.sign(w0))*sym.sqrt(sym.Abs(w0)/2)
    }
    return u0_sph, pars


def magbl_correction(u0, par0, B0, nSH_V=30, verbose=False):

    m_val = par0[iw.m]
    f_u0_mod_sq = sym.lambdify([r, theta], dot(conj(u0), u0).subs(par0), modules=['numpy', 'scipy'])
    u0_l2 = iw.quad_ball_axisym(f_u0_mod_sq)
    
    # Interior field
    order = 0
    b1_in = sph.curl_m(cross(u0, B0), m_val)
    for i_order in range(order):
        b1_in = -sph.curl_m(sph.curl_m(b1_in), m_val)
    b1_in = [bi/(iw.elamb**(int(order) + 1)) for bi in b1_in]

    # Interior fields at boundary
    u0_b = [ui.subs({r: 1}) for ui in u0]
    b1_in_b = [bi.subs({r: 1}) for bi in b1_in]
    f_b1r = sym.lambdify([theta,], b1_in_b[0].subs(par0), modules=['numpy', 'scipy'])
    f_b1t_b_int = sym.lambdify([theta,], b1_in_b[1].subs(par0), modules=['numpy', 'scipy'])
    f_b1p_b_int = sym.lambdify([theta,], b1_in_b[2].subs(par0), modules=['numpy', 'scipy'])

    # 1st-order boundary layer solutions
    f_b1t_b, f_b1p_b, cf = iw.rad2tan_B_single_m(f_b1r, nSH_V, m_val)
    f_b1t_bl = lambda t: f_b1t_b(t) - f_b1t_b_int(t)
    f_b1p_bl = lambda t: f_b1p_b(t) - f_b1p_b_int(t)

    # 2nd-order boundary layer solutions
    dr_r2_b1r_in_b = sym.diff(r**2*b1_in[0], r).subs({r: 1})
    f_dr_r2_b1r_in_b = sym.lambdify([theta,], dr_r2_b1r_in_b.subs(par0), modules=['numpy', 'scipy'])

    def f_b1hr_bl0(t):
        xi_t = np.cos(t)
        basis = specfun.assoc_legendre_p_all(nSH_V, m_val, xi_t)[0, m_val:, m_val].T
        n_arr = np.arange(m_val, nSH_V + 1)
        return (-1/complex(par0[alpha]))*(basis @ (cf*n_arr*(n_arr + 1)) + f_dr_r2_b1r_in_b(t))

    f_b1ht_bl0, f_b1hp_bl0, cfh = iw.rad2tan_B_single_m(f_b1hr_bl0, nSH_V, m_val)

    # Eigenvalue perturbations
    # Interior
    L_in_A = cross(sph.curl_m(b1_in, m_val), B0)
    L_in_B = cross(sph.curl_m(B0, 0), b1_in)
    L_in = [(L_in_A[i] + L_in_B[i]) for i in range(3)]
    f_uL_in = sym.lambdify([r, theta], dot(conj(u0), L_in).subs(par0), modules=['numpy', 'scipy'])
    inner_u_L_in = iw.quad_ball_axisym(f_uL_in)

    # 1st-order BL
    f_u0t_b = sym.lambdify([theta,], u0_b[1].subs(par0), modules=['numpy', 'scipy'])
    f_u0p_b = sym.lambdify([theta,], u0_b[2].subs(par0), modules=['numpy', 'scipy'])
    f_B0r = sym.lambdify([theta,], B0[0].subs({r: 1}), modules=['numpy', 'scipy'])
    f_surf = lambda t: f_B0r(t)*(
        + np.conj(f_u0t_b(t))*f_b1t_bl(t)
        + np.conj(f_u0p_b(t))*f_b1p_bl(t)
    )
    inner_u_L_bl = iw.quad_sph_surf_axisym(f_surf)

    # 2nd-order BL
    psi_0 = [cross(u0, sph.curl_m(B0, 0)), sph.curl_m(cross(u0, B0), m_val)]
    psi_0 = [psi_0[0][i_comp] - psi_0[1][i_comp] for i_comp in range(3)]
    f_psi_0t = sym.lambdify([theta,], psi_0[1].subs({r: 1, **par0}), modules=['numpy', 'scipy'])
    f_psi_0p = sym.lambdify([theta,], psi_0[2].subs({r: 1, **par0}), modules=['numpy', 'scipy'])
    f_surf_1 = lambda t: f_B0r(t)*(
        + np.conj(f_u0t_b(t))*f_b1ht_bl0(t)
        + np.conj(f_u0p_b(t))*f_b1hp_bl0(t)
    )
    f_surf_2 = lambda t: (
        + np.conj(f_psi_0t(t))*f_b1t_bl(t)
        + np.conj(f_psi_0p(t))*f_b1p_bl(t)
    )/complex(par0[alpha])
    inner_u_L_bl2_1 = iw.quad_sph_surf_axisym(f_surf_1)
    inner_u_L_bl2_2 = iw.quad_sph_surf_axisym(f_surf_2)

    # Calculate eigenvalue perturbations
    ev_2_0_in = inner_u_L_in/u0_l2
    ev_2_0_bl = inner_u_L_bl/u0_l2
    ev_2_0 = ev_2_0_in + ev_2_0_bl
    ev_2_h_blh = inner_u_L_bl2_1/u0_l2
    ev_2_h_bl0 = inner_u_L_bl2_2/u0_l2
    ev_2_h = ev_2_h_blh + ev_2_h_bl0

    results = {
        'ev_Le2_Em0': ev_2_0,
        'ev_Le2_Em0_in': ev_2_0_in,
        'ev_Le2_Em0_bl': ev_2_0_bl,
        'VSH_Le2_Em0': cf,
        'ev_Le2_Emh': ev_2_h,
        'ev_Le2_Emh_bl0': ev_2_h_bl0,
        'ev_Le2_Emh_blh': ev_2_h_blh,
        'VSH_Le2_Emh': cfh,
    }
    return results


B0_cyl_lib = {
    'U': (S.Zero, S.Zero, S.One),
    'S1': (3*s*z/5, S.Zero, (5 - 6*s**2 - 3*z**2)/5),
    'T2': (S.Zero, 8*s*z*(1 - s**2 - z**2), S.Zero)
}

B0_sph_lib = {
    key: cyl2sph_vec(B0, simplify=True)
    for key, B0 in B0_cyl_lib.items()
}
