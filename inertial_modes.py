# -*- coding: utf-8 -*-


import numpy as np
import sympy as sym
import gmpy2 as gp
from scipy import special as specfun
from typing import Union, Optional


m, N, i, j = sym.symbols('m,N,i,j', integer=True)
s, r, theta = sym.symbols(r's,r,\theta', positive=True)
phi, z = sym.symbols(r'\phi, z', real=True)
sigma = sym.Symbol(r'\sigma')
omega = sym.Symbol(r'\omega')
elamb = sym.Symbol(r'\lambda')


eigenfreq_poly_terms = {
    's': (
        (-1)**j*sym.factorial(2*(2*N + m - j))
        /(sym.factorial(j)*sym.factorial(2*N + m - j)*sym.factorial(2*(N-j)))
        *((2*N + m - 2*j)*sigma - 2*(N - j))*sigma**(2*(N - j)-1)
    ),
    'a': (
        (-1)**j*sym.factorial(2*(2*N + m - j + 1))
        /(sym.factorial(j)*sym.factorial(2*N + m - j + 1)*sym.factorial(2*(N-j) + 1))
        *((2*N + m - 2*j + 1)*sigma - (2*(N - j) + 1))*sigma**(2*(N - j))
    )
}
eigenfreq_polys = {
    's': sym.Sum(eigenfreq_poly_terms['s'], (j, 0, N)),
    'a': sym.Sum(eigenfreq_poly_terms['a'], (j, 0, N)),
}


mode_poly_cfs = {
    's': (
        (-1)**(i + j)*sym.factorial2(2*(N + m + i + j) - 1)
        /(2**(j + 1)*sym.factorial2(2*i - 1))
        /(sym.factorial(N - i - j)*sym.factorial(i)*sym.factorial(j)*sym.factorial(m + j))
    ),
    'a': (
        (-1)**(i + j)*sym.factorial2(2*(N + m + i + j) + 1)
        /(2**(j + 1)*sym.factorial2(2*i + 1))
        /(sym.factorial(N - i - j)*sym.factorial(i)*sym.factorial(j)*sym.factorial(m + j))
    )
}
mode_poly_terms_cyl = {
    's': [
        (m + (2*j + m)*sigma)*sigma**(2*i) * (1 - sigma**2)**(j - 1) * s**(m + 2*j - 1) * z**(2*i),
        (m + 2*j + m*sigma)*sigma**(2*i) * (1 - sigma**2)**(j - 1) * s**(m + 2*j - 1) * z**(2*i),
        2*i*sigma**(2*i - 1) * (1 - sigma**2)**j * s**(m + 2*j) * z**(2*i - 1),
    ],
    'a': [
        (m + (2*j + m)*sigma)*sigma**(2*i) * (1 - sigma**2)**(j - 1) * s**(m + 2*j - 1) * z**(2*i+1),
        (m + 2*j + m*sigma)*sigma**(2*i) * (1 - sigma**2)**(j - 1) * s**(m + 2*j - 1) * z**(2*i + 1),
        (2*i + 1)*sigma**(2*i - 1) * (1 - sigma**2)**j * s**(m + 2*j) * z**(2*i),
    ]
}
mode_p_terms_cyl = {
    's': sigma**(2*i) * (1 - sigma**2)**j * s**(m + 2*j) * z**(2*i),
    'a': sigma**(2*i) * (1 - sigma**2)**j * s**(m + 2*j) * z**(2*i + 1)
}


def eigenfreq_inviscid(N_val, m_val, parity='s', sort=True, filter_zero=True, **solve_kwargs):
    """Calculate eigenfrequencies of the inviscid inertial modes in unit sphere
    """
    poly = eigenfreq_polys[parity]
    poly = poly.subs({N: N_val, m: m_val}).doit()
    roots = sym.nroots(poly, **solve_kwargs)
    eigenfreqs = 2*np.array(roots)
    eigenfreqs = eigenfreqs[np.argsort(np.abs(eigenfreqs))]
    eigenfreqs = eigenfreqs[np.abs(eigenfreqs) > 1e-7]
    return eigenfreqs


def which_eigenfreq(freq_0, Ns, ms):
    """
    """
    if isinstance(Ns, int):
        Ns = np.arange(1, Ns)
    Ns = np.atleast_1d(np.asarray(Ns))
    ms = np.atleast_1d(np.asarray(ms))
    m, n, k = 0, 0, 0
    parity = 's'
    freq = 0.
    r_diff = 100.
    
    for m_tmp in ms:
        for n_tmp in Ns:
            for par in ['s', 'a']:
                
                freqs = eigenfreq_inviscid(n_tmp, m_tmp, parity=par)
                r_diff_tmp = np.abs(freqs - freq_0)/np.abs(freq_0)
                k_tmp = np.argmin(r_diff_tmp)
                r_diff_tmp = r_diff_tmp[k_tmp]
                
                if r_diff_tmp >= r_diff:
                    continue
                
                r_diff = r_diff_tmp
                m, n, k = m_tmp, n_tmp, k_tmp
                parity = par
                freq = freqs[k_tmp]
    
    return m, n, k, parity, freq


def code_convert(m, n, k, X=None):
    if X is None:
        return m, (n - m) // 2, k, 'S' if (n - m)%2 == 0 else 1
    else:
        iX = 0 if X == 'S' else 1
        return m, m + 2*n + iX, k

def eigenmode_poly_inviscid(N_val, parity='s'):
    u_s = -sym.I*sym.Add(*[
        (mode_poly_cfs[parity]*mode_poly_terms_cyl[parity][0]).subs({N: N_val, i: i_val, j: j_val}) 
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    u_p = sym.Add(*[
        (mode_poly_cfs[parity]*mode_poly_terms_cyl[parity][1]).subs({N: N_val, i: i_val, j: j_val}) 
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    u_z = +sym.I*sym.Add(*[
        (mode_poly_cfs[parity]*mode_poly_terms_cyl[parity][2]).subs({N: N_val, i: i_val, j: j_val}) 
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    return u_s, u_p, u_z


def pressure_mode_invisicid(N_val, parity='s'):
    p_mode = sym.Add(*[
        (mode_poly_cfs[parity]*mode_p_terms_cyl[parity]).subs({N: N_val, i: i_val, j: j_val})
        for i_val in range(N_val + 1) for j_val in range(N_val - i_val + 1)
    ])
    return p_mode


def transform_SH(vr_func, Ntrunc: int, m_val: int): 
    """
    """
    xi, _ = specfun.roots_chebyt(2*Ntrunc)
    # Pmn_vals = [specfun.assoc_legendre_p_all(m_val, Ntrunc, xi_tmp) for xi_tmp in xi]
    # Pmn = np.stack([Pmn_tmp[0][m_val:, m_val] for Pmn_tmp in Pmn_vals], axis=0)
    Pmn = specfun.assoc_legendre_p_all(Ntrunc, m_val, xi)[0, m_val:, m_val, :].T
    vr_vals = vr_func(np.arccos(xi))*np.ones_like(xi)
    c_SH = np.linalg.solve(Pmn.T @ Pmn, Pmn.T @ vr_vals)
    return c_SH


def rad2tan_B_single_m(Br_func, Ntrunc: int, m_val: int):
    """
    """
    xi, wt = specfun.roots_chebyt(2*Ntrunc)
    # Pmn_vals = [specfun.assoc_legendre_p_all(m_val, Ntrunc, xi_tmp) for xi_tmp in xi]
    # Pmn = np.stack([Pmn_tmp[0][m_val:, m_val] for Pmn_tmp in Pmn_vals], axis=0)
    Pmn = specfun.assoc_legendre_p_all(Ntrunc, m_val, xi)[0, m_val:, m_val, :].T
    Br_vals = Br_func(np.arccos(xi))*np.ones_like(xi)
    cf = np.linalg.solve(Pmn.T @ Pmn, Pmn.T @ Br_vals)
    
    cf_gauss = cf/(np.arange(m_val, Ntrunc + 1) + 1)
    def B_t(t):
        xi_t = np.cos(t)
        # basis = np.stack([specfun.assoc_legendre_p_all(m_val, Ntrunc, xi_tmp)[1][m_val:, m_val] for xi_tmp in xi_t], axis=0)
        basis = specfun.assoc_legendre_p_all(Ntrunc, m_val, xi_t, diff_n=1)[1, m_val:, m_val, :].T
        return np.sin(t)*(basis @ cf_gauss)
    
    def B_p(t):
        xi_t = np.cos(t)
        # basis = np.stack([specfun.assoc_legendre_p_all(m_val, Ntrunc, xi_tmp)[0][m_val:, m_val] for xi_tmp in xi_t], axis=0)
        basis = specfun.assoc_legendre_p_all(Ntrunc, m_val, xi_t)[0, m_val:, m_val, :].T
        return -(1j*m_val/np.sin(t))*(basis @ cf_gauss)
    
    return B_t, B_p, cf_gauss


def int_sph_surf_axisym(integrand, rad=sym.S.One):
    integral = 2*sym.pi*rad**2*sym.integrate(
        integrand.subs({r: rad})*sym.sin(theta), 
        (theta, sym.S.Zero, sym.pi)
    )
    return integral


def quad_sph_surf_axisym(f_integrand, rad=sym.S.One, N=101):
    # f_integrand = sym.lambdify((theta,), integrand.subs({r: rad}), modules=['numpy', 'scipy'])
    xi_quad, wt_quad = specfun.roots_legendre(N)
    t_quad = np.arccos(xi_quad)
    return 2*np.pi*float(rad)**2*(wt_quad @ (f_integrand(t_quad)*np.ones_like(t_quad)))


def int_ball_axisym(integrand, rad=sym.S.One):
    integral = 2*sym.pi*sym.integrate(
        integrand*r**2*sym.sin(theta), 
        (theta, sym.S.Zero, sym.pi), 
        (r, sym.S.Zero, rad)
    )
    return integral


def quad_ball_axisym(f_integrand, rad=sym.S.One, Nt=101, Nr=101):
    xi_quad, wt_r = specfun.roots_legendre(Nr)
    r_quad, wt_r = (1 + xi_quad)/2, wt_r/2
    xi_quad, wt_t = specfun.roots_legendre(Nt)
    t_quad = np.arccos(xi_quad)
    rr, tt = np.meshgrid(r_quad, t_quad, indexing='ij')
    integral = 2*np.pi*(wt_r*r_quad**2) @ ((f_integrand(rr, tt)*np.ones_like(tt)) @ wt_t)
    return integral


def eigenfreq_Rossby_to_Malkus(
    m, omega_0, Le: Union[float, gp.mpfr], 
    timescale: str="spin", prec: Optional[int] = None
):
    """Analytic eigenfrequency for the PG model with Malkus bg field
    
    :param Union[int, np.ndarray] m: azimuthal wavenumber
    :param omega_0: inertial mode eigenfrequencies
    :param float Le: Lehnert number (see also :data:`~pg_utils.pg_model.params.Le` )
    :param str mode: fast or slow, default to "all"
    :param str timescale: characteristic timescale, default to "spin", 
        alternative: "alfven". See note below for more details.
    :param Optional[int] prec: precision to be computed to. 
        Default to None, calculate to double precision using numpy.
    
    :returns: eigenfrequency array(s)
    
    .. note::
    
        When using spin rate for characteristic time scale, i.e. :math:`\\tau=\\Omega^{-1}`
        
        .. math::

            \\omega = \\frac{\\omega_0}{2} 
            \\left(1 \\pm \\sqrt{\\mathrm{Le}^2 \\frac{4m(m - \\omega_0)}{\\omega_0^2}}\\right)
        
        When using Alfven time scale, i.e. :math:`\\tau=\\frac{\\sqrt{\\rho\\mu_0}L}{B}`
        
        .. math::
        
            \\omega = \\frac{\\omega_0}{2\\mathrm{Le}} 
            \\left(1 \\pm \\sqrt{\\mathrm{Le}^2 \\frac{4m(m - \\omega_0)}{\\omega_0^2}}\\right)
        
        where :math:`\\omega_0` is the inertial mode eigenfrequency. 
        The plus sign gives the fast mode, and the minus sign gives the slow mode.
    """
    if prec is None:
        if timescale.lower() == "spin":
            prefactor = omega_0/2
        elif timescale.lower() == "alfven":
            prefactor = omega_0/2/Le
        else:
            raise AttributeError
        bg_field_mod = np.sqrt(1 + Le**2*(4*m*(m - omega_0))/(omega_0**2))
        return prefactor*(1 + bg_field_mod), prefactor*(1 - bg_field_mod)
    else:
        with gp.local_context(gp.context(), precision=prec):
            if timescale.lower() == "spin":
                prefactor = omega_0/2
            elif timescale.lower() == "alfven":
                prefactor = omega_0/2/Le
            else:
                raise AttributeError
            bg_field_mod = 1 + Le**2*(4*m*(m - omega_0))/(omega_0**2)
            bg_field_mod = np.vectorize(gp.sqrt, otypes=(object,))(bg_field_mod)
            return prefactor*(1 + bg_field_mod), prefactor*(1 - bg_field_mod)


if __name__ == '__main__':
    
    # eigenfreqs = eigenfreq_inviscid(1, 3, parity='a')
    # eigenfreqs = np.array([eigenfreq_inviscid(n, 3, parity='a')[0] for n in range(10)])
    eigenfreqs = np.array([eigenfreq_inviscid(n, 3, parity='s')[1] for n in range(1, 10)])
    print(eigenfreqs)
