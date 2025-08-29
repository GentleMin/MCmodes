"""Module provides functions to generate values for the Worland expansion"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import scipy.special as special

import quicc.base.utils as utils

def worland_norm(n , l):
    """Normalization factor = 1/Norm
    """

    if l == 0:
        if n == 0:
            return np.sqrt(2.0/np.pi)
        else:
            return 2.0*np.exp(special.gammaln(n+1.0) - special.gammaln(n+0.5))
    else:
        return np.sqrt(2.0*(2.0*n+l)*np.exp(special.gammaln(n+l) + special.gammaln(n+1.0) - special.gammaln(n+0.5) - special.gammaln(n+l+0.5)))

def worland_grid(nr):
    """Physical space grid"""

    return np.sqrt((np.cos(np.pi*(np.arange(0,2*nr)+0.5)/(2*nr)) + 1.0)/2.0)

def worland_weight(i, nr):
    """Gaussian integration weight"""

    return np.pi/(2.0*nr)

def worland_poly(n, l, nr):

    rg = worland_grid(nr)
    norm = worland_norm(n, l)

    return rg**l*special.eval_jacobi(n, -0.5, l - 0.5, 2.0*rg**2 - 1.0)*norm

def worland_norm_row(n, l, k, p = 0):
    """Normalization factor for matrix row. l is from LHS, p allows to shift RHS to l+p"""

    if n[0] + k < 0:
        raise RuntimeError("Requested row normalization is inconsistent")

    if l == 0:
        norm = -np.log(4.0) + 2.0*special.gammaln(n + 0.5) - 2.0*special.gammaln(n + 1.0)
        if n[0] == 0:
            norm[0] = (np.log(np.pi) - np.log(2.0))
    else:
        norm = -np.log(2.0*(2.0*n + l)) + special.gammaln(n + 0.5) + special.gammaln(n + l + 0.5) - special.gammaln(n + l) - special.gammaln(n + 1.0)

    # Special case for projection on l = 0
    if l + p == 0:
        if n[0] + k == 0:
            norm[0] += np.log(2.0) - np.log(np.pi)
            norm[1:] += np.log(4.0) - 2.0*special.gammaln(n[1:] + 0.5 + k) + 2.0*special.gammaln(n[1:] + 1.0 + k)
        else:
            norm += np.log(4.0) - 2.0*special.gammaln(n + 0.5 + k) + 2.0*special.gammaln(n + 1.0 + k)
    else:
        norm += np.log(2.0*(2.0*n + l + p + 2.0*k)) - special.gammaln(n + 0.5 + k) - special.gammaln(n + l + p + 0.5 + k) + special.gammaln(n + l + p + k) + special.gammaln(n + 1.0 + k)

    return np.sqrt(np.exp(norm))

def worland_norm_row_l_1(n, l, k):
    """Normalization factor for matrix row from W_n^l to Wn^{l-1}"""

    if n[0] + k < 0:
        raise RuntimeError("Requested row normalization is inconsistent")
  
    if l - 1 == 0:
        norm = -np.log(4.0) + 2.0*special.gammaln(n + 0.5) - 2.0*special.gammaln(n + 1.0)
        if n[0] == 0:
            norm[0] = (np.log(np.pi) - np.log(2.0))
    elif l == 0:
        norm = -np.log(2.0*(2.0*n + l + 1.0)) + special.gammaln(n + 0.5) + special.gammaln(n + l + 1.5) - special.gammaln(n + l + 1.0) - special.gammaln(n + 1.0)
    else:
        norm = -np.log(2.0*(2.0*n + l - 1.0)) + special.gammaln(n + 0.5) + special.gammaln(n + l - 0.5) - special.gammaln(n + l - 1.0) - special.gammaln(n + 1.0)

    # Special case for projection on l = 0
    if l == 0:
        if n[0] + k == 0:
            norm[0] += np.log(2.0) - np.log(np.pi)
            norm[1:] += np.log(4.0) - 2.0*special.gammaln(n[1:] + 0.5 + k) + 2.0*special.gammaln(n[1:] + 1.0 + k)
        else:
            norm += np.log(4.0) - 2.0*special.gammaln(n + 0.5 + k) + 2.0*special.gammaln(n + 1.0 + k)
    else:
        norm += np.log(2.0*(2.0*n + l + 2.0*k)) - special.gammaln(n + 0.5 + k) - special.gammaln(n + l + 0.5 + k) + special.gammaln(n + l + k) + special.gammaln(n + 1.0 + k)

    return np.sqrt(np.exp(norm))

def worland_normalize(val, l):
    """Normalize array of values"""

    for i in range(0, val.shape[0]):
         val[i] *= worland_norm(i, l)

def worland_origin(nr, l, k = 0, normalize = True):
    """Compute the value at origin for Worland polynomials"""

    val = np.zeros(nr)
    val[0] = 1.0
    if nr > 0:
        for i in range(1,nr):
            val[i] = (-1.0)**i*np.exp(special.gammaln(i + l + 0.5) - special.gammaln(i + 1.0) - special.gammaln(l + 0.5))

    # Normalize
    if normalize:
        worland_normalize(val, l)

    return val

def worland_value(nr, l, k = 0, normalize = True):
    """Compute the endpoint value for Worland polynomials"""

    val = np.zeros(nr)
    val[0] = 1.0
    if nr > 0:
        for i in range(1,nr):
            val[i] = val[i-1]*(2.0*i-1.0 + 2*k)/(2.0*i)

    # Normalize
    if normalize:
        worland_normalize(val, l)

    return val

def worland_diff(nr, l):
    """Compute the first derivative at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[1:] = 2.0*(l+np.arange(1,nr))*worland_value(nr-1, l+1, 1, False)
        if l > 0:
            val += l*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_diff2(nr, l):
    """Compute the second derivative at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[2:] = 4.0*(l+np.arange(2,nr)+1)*(l+np.arange(2,nr))*worland_value(nr-2, l+2, 2, False)
        if l > 0:
            val[1:] += (2.0*(l+np.arange(1,nr)) + 4.0*l*(l+np.arange(1,nr)))*worland_value(nr-1, l+1, 1, False)
        else:
            val[1:] += 2.0*(l+np.arange(1,nr))*worland_value(nr-1, l+1, 1, False)
        if l > 1:
            val += l*(l-1.0)*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_diff3(nr, l):
    """Compute the third derivative at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[3:] = 8.0*(l+np.arange(3,nr)+2)*(l+np.arange(3,nr)+1)*(l+np.arange(3,nr))*worland_value(nr-3, l+3, 3, False)
        val[2:] += 12.0*(l+np.arange(2,nr))*(l+np.arange(2,nr)+1.0)*(1.0 + l)*worland_value(nr-2, l+2, 2, False)
        if l > 0:
            val[1:] += 6.0*l*(l+np.arange(1,nr))*l*worland_value(nr-1, l+1, 1, False)
        if l > 2:
            val += (l-1.0)*(l-2.0)*l*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_diff4(nr, l):
    """Compute the fourth derivative at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[4:] = 16.0*(l+np.arange(4,nr)+3)*(l+np.arange(4,nr)+2)*(l+np.arange(4,nr)+1)*(l+np.arange(4,nr))*worland_value(nr-4, l+4, 4, False)
        val[3:] += 16.0*(2.0*l + 3.0)*(l+np.arange(3,nr))*(l+np.arange(3,nr)+1.0)*(l+np.arange(3,nr)+2.0)*worland_value(nr-3, l+3, 3, False)
        val[2:] += 12.0*(2.0*l**2 + 2.0*l + 1.0)*(l+np.arange(2,nr))*(l+np.arange(2,nr)+1)*worland_value(nr-2, l+2, 2, False)
        if l > 1:
            val[1:] += 4.0*(l-1.0)*l*(2.0*l-1.0)*(l+np.arange(1,nr))*worland_value(nr-1, l, 1, False)
        if l > 3:
            val += l*(l-1.0)*(l-2.0)*(l-3.0)*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_rdiffdivr(nr, l):
    """Compute the stress-free condition at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[1:] = 2.0*(l+np.arange(1,nr))*worland_value(nr-1, l+1, 1, False)
        val += (l-1.0)*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_divrdiffr(nr, l):
    """Compute the 1/r D r condition at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[1:] = 2.0*(l+np.arange(1,nr))*worland_value(nr-1, l+1, 1, False)
        val += (l+1.0)*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_insulating_sph(nr, l):
    """Compute the insulating magnetic condition for a sphere at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[1:] = 2.0*(l+np.arange(1,nr))*worland_value(nr-1, l+1, 1, False)
        val += (2.0*l+1.0)*worland_value(nr, l, 0, False)

    # Normalize
    worland_normalize(val, l)

    return val

def worland_laplh_cyl(nr, m):
    """Compute the horizontal laplacian in a cylinder at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[2:] = 4.0*(m+np.arange(2,nr)+1)*(m+np.arange(2,nr))*worland_value(nr-2, m+2, 2, False)
        val[1:] += 4.0*(m+1.0)*(m+np.arange(1,nr))*worland_value(nr-1, m+1, 1, False)

    # Normalize
    worland_normalize(val, m)

    return val

def worland_dlaplh_cyl(nr, m):
    """Compute the radial derivative of horizontal laplacian in a cylinder at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[3:] = 8.0*(m+np.arange(3,nr)+2)*(m+np.arange(3,nr)+1)*(m+np.arange(3,nr))*worland_value(nr-3, m+3, 3, False)
        val[2:] += 4.0*(3.0*m + 4.0)*(m+np.arange(2,nr)+1)*(m+np.arange(2,nr))*worland_value(nr-2, m+2, 2, False)
        if m > 0:
            val[1:] += 4.0*m*(m + 1.0)*(m+np.arange(1,nr))*worland_value(nr-1, m+1, 1, False)

    # Normalize
    worland_normalize(val, m)

    return val

def worland_lapl2h_cyl(nr, m):
    """Compute the bilaplacian in a cylinder at endpoint for Worland polynomials"""

    val = np.zeros(nr)
    if nr > 0:
        val[4:] = 16.0*(m+np.arange(4,nr)+3)*(m+np.arange(4,nr)+2)*(m+np.arange(4,nr)+1)*(m+np.arange(4,nr))*worland_value(nr-4, m+4, 4, False)
        val[3:] += 32.0*(m + 2.0)*(m+np.arange(3,nr)+2)*(m+np.arange(3,nr)+1)*(m+np.arange(3,nr))*worland_value(nr-3, m+3, 3, False)
        val[2:] += 16.0*(m + 1.0)*(m + 2.0)*(m+np.arange(2,nr)+1)*(m+np.arange(2,nr))*worland_value(nr-2, m+2, 2, False)

    # Normalize
    worland_normalize(val, m)

    return val

def i1_diags(nr, l):
    """Create operator for 1st integral r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-1,2)
    nzrow = 0

    # Generate 1st subdiagonal
    def d_1(n):
        val = worland_norm_row(n,l,-1)*2.0*(l + n - 1.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))
        if l == 0:
            val[0] = worland_norm_row(n[0:1],l,-1)*2.0/(l + 1.0)
        return val

    # Generate main diagonal
    def d0(n):
        return -worland_norm_row(n,l,0)*2.0*l/((l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -worland_norm_row(n,l,1)*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)/(2.0*(l + n)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    ds = [d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)

def i2_diags(nr, l):
    """Create operator for 2nd integral r^l P_n^{-1/2,l-1/2}(2r^2 -1)."""

    ns = np.arange(0, nr+1)
    offsets = np.arange(-2,3)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        if l == 0:
            val = worland_norm_row(n,l,-2)/((l + 2.0*n - 3.0)*(l + 2.0*n - 1.0))
            val[0] = worland_norm_row(n[0:1],l,-2)*4.0*(l + 1.0)/((l + 1.0)*(l + 2.0)*(l + 3.0))
        else:
            val = worland_norm_row(n,l,-2)*4.0*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))
        return val

    # Generate 1st subdiagonal
    def d_1(n):
        return -worland_norm_row(n,l,-1)*8.0*l*(l + n - 1.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate diagonal
    def d0(n):
        return worland_norm_row(n,l,0)*2.0*(2.0*l**2 - 4.0*l*n - 4.0*n**2 + 1.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return worland_norm_row(n,l,1)*2.0*l*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)/((l + n)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return worland_norm_row(n,l,2)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)/(4.0*(l + n)*(l + n + 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)

def i4_diags(nr, l):
    """Create operator for 4th integral r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-4,5)
    nzrow = 3

    # Generate 4th subdiagonal
    def d_4(n):
        if l == 0:
            val = worland_norm_row(n,l,-4)/((l + 2.0*n - 7.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 1.0))
            val[0] = worland_norm_row(n[0:1],l,-4)*16.0/((l + 4.0)*(l + 5.0)*(l + 6.0)*(l + 7.0))
        else:
            val = worland_norm_row(n,l,-4)*16.0*(l + n - 4.0)*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 8.0)*(l + 2.0*n - 7.0)*(l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))
        return val

    # Generate 3rd subdiagonal
    def d_3(n):
        return -worland_norm_row(n,l,-3)*64.0*l*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 7.0)*(l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return worland_norm_row(n,l,-2)*16.0*(l + n - 2.0)*(l + n - 1.0)*(6.0*l**2 - 4.0*l*n + 4.0*l - 4.0*n**2 + 8.0*n + 5.0)/((l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -worland_norm_row(n,l,-1)*16.0*l*(l + n - 1.0)*(4.0*l**2 - 12.0*l*n + 6.0*l - 12.0*n**2 + 12.0*n + 17.0)/((l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate diagonal
    def d0(n):
        return worland_norm_row(n,l,0)*2.0*(8.0*l**4 - 96.0*l**3*n - 48.0*l**2*n**2 + 100.0*l**2 + 96.0*l*n**3 - 120.0*l*n + 48.0*n**4 - 120.0*n**2 + 27.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    # Generate 1st superdiagonal
    def d1(n):
        return worland_norm_row(n,l,1)*4.0*l*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)*(4.0*l**2 - 12.0*l*n - 6.0*l - 12.0*n**2 - 12.0*n + 17.0)/((l + n)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return worland_norm_row(n,l,2)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(6.0*l**2 - 4.0*l*n - 4.0*l - 4.0*n**2 - 8.0*n + 5.0)/((l + n)*(l + n + 1.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return worland_norm_row(n,l,3)*l*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)/((l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0)*(l + 2.0*n + 7.0))

    # Generate 4th superdiagonal
    def d4(n):
        return worland_norm_row(n,l,4)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)*(2.0*l + 2.0*n + 7.0)/(16.0*(l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + n + 3.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0)*(l + 2.0*n + 7.0)*(l + 2.0*n + 8.0))

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)

def i6_diags(nr, m):
    """Create operator for 6th integral r^m P_n^{-1/2,m-1/2}(2r^2-1)."""

    ns = np.arange(0, nr+3)
    offsets = np.arange(-6,7)
    nzrow = 5

    # Generate 6th subdiagonal
    def d_6(n):
        if m == 0:
            val = worland_norm_row(n, m, -6)/((m + 2.0*n - 11.0)*(m + 2.0*n - 9.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 1.0))
            val[0] = worland_norm_row(n[0:1], m, -6)*2.0/10395.0
        else:
            val = worland_norm_row(n, m, -6)*64.0*(m + n - 6.0)*(m + n - 5.0)*(m + n - 4.0)*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)/((m + 2.0*n - 12.0)*(m + 2.0*n - 11.0)*(m + 2.0*n - 10.0)*(m + 2.0*n - 9.0)*(m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0))
        return val

    # Generate 5th subdiagonal
    def d_5(n):
        return -worland_norm_row(n, m, -5)*384.0*m*(m + n - 5.0)*(m + n - 4.0)*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)/((m + 2.0*n - 11.0)*(m + 2.0*n - 10.0)*(m + 2.0*n - 9.0)*(m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return worland_norm_row(n, m, -4)*96.0*(m + n - 4.0)*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)*(10.0*m**2 - 4.0*m*n + 8.0*m - 4.0*n**2 + 16.0*n + 9.0)/((m + 2.0*n - 10.0)*(m + 2.0*n - 9.0)*(m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -worland_norm_row(n, m, -3)*160.0*m*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)*(8.0*m**2 - 12.0*m*n + 18.0*m - 12.0*n**2 + 36.0*n + 37.0)/((m + 2.0*n - 9.0)*(m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return worland_norm_row(n, m, -2)*60.0*(m + n - 2.0)*(m + n - 1.0)*(16.0*m**4 - 64.0*m**3*n + 64.0*m**3 - 48.0*m**2*n**2 + 96.0*m**2*n + 236.0*m**2 + 32.0*m*n**3 - 96.0*m*n**2 - 40.0*m*n + 104.0*m + 16.0*n**4 - 64.0*n**3 - 40.0*n**2 + 208.0*n + 105.0)/((m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -worland_norm_row(n, m, -1)*48.0*m*(m + n - 1.0)*(8.0*m**4 - 80.0*m**3*n + 40.0*m**3 + 280.0*m**2 + 160.0*m*n**3 - 240.0*m*n**2 - 440.0*m*n + 260.0*m + 80.0*n**4 - 160.0*n**3 - 440.0*n**2 + 520.0*n + 537.0)/((m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0))

    # Generate diagonal
    def d0(n):
        return worland_norm_row(n, m, 0)*4.0*(16.0*m**6 - 480.0*m**5*n + 960.0*m**4*n**2 + 1120.0*m**4 + 2560.0*m**3*n**3 - 7840.0*m**3*n + 480.0*m**2*n**4 - 5040.0*m**2*n**2 + 5614.0*m**2 - 960.0*m*n**5 + 5600.0*m*n**3 - 5180.0*m*n - 320.0*n**6 + 2800.0*n**4 - 5180.0*n**2 + 1125.0)/((m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0))

    # Generate 1st superdiagonal
    def d1(n):
        return worland_norm_row(n, m, 1)*12.0*m*(2.0*n + 1.0)*(2.0*m + 2.0*n + 1.0)*(8.0*m**4 - 80.0*m**3*n - 40.0*m**3 + 280.0*m**2 + 160.0*m*n**3 + 240.0*m*n**2 - 440.0*m*n - 260.0*m + 80.0*n**4 + 160.0*n**3 - 440.0*n**2 - 520.0*n + 537.0)/((m + n)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return worland_norm_row(n, m, 2)*15.0*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(16.0*m**4 - 64.0*m**3*n - 64.0*m**3 - 48.0*m**2*n**2 - 96.0*m**2*n + 236.0*m**2 + 32.0*m*n**3 + 96.0*m*n**2 - 40.0*m*n - 104.0*m + 16.0*n**4 + 64.0*n**3 - 40.0*n**2 - 208.0*n + 105.0)/(4.0*(m + n)*(m + n + 1.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return worland_norm_row(n, m, 3)*5.0*m*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(8.0*m**2 - 12.0*m*n - 18.0*m - 12.0*n**2 - 36.0*n + 37.0)/(2.0*(m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0)*(m + 2.0*n + 9.0))

    # Generate 4th superdiagonal
    def d4(n):
        return worland_norm_row(n, m, 4)*3.0*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(2.0*m + 2.0*n + 7.0)*(10.0*m**2 - 4.0*m*n - 8.0*m - 4.0*n**2 - 16.0*n + 9.0)/(8.0*(m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + n + 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0)*(m + 2.0*n + 9.0)*(m + 2.0*n + 10.0))

    # Generate 5rd superdiagonal
    def d5(n):
        return worland_norm_row(n, m, 5)*3.0*m*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*n + 9.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(2.0*m + 2.0*n + 7.0)*(2.0*m + 2.0*n + 9.0)/(8.0*(m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + n + 3.0)*(m + n + 4.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0)*(m + 2.0*n + 9.0)*(m + 2.0*n + 10.0)*(m + 2.0*n + 11.0))

    # Generate 6th superdiagonal
    def d6(n):
        return worland_norm_row(n, m, 6)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*n + 9.0)*(2.0*n + 11.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(2.0*m + 2.0*n + 7.0)*(2.0*m + 2.0*n + 9.0)*(2.0*m + 2.0*n + 11.0)/(64.0*(m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + n + 3.0)*(m + n + 4.0)*(m + n + 5.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0)*(m + 2.0*n + 9.0)*(m + 2.0*n + 10.0)*(m + 2.0*n + 11.0)*(m + 2.0*n + 12.0))

    ds = [d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)

#######################################################
# OPERATOR BELOW THIS LINE HAVE NOT BEEN CHECKED

def i2r_1d1_diags(nr, l):
    """Create operator for 2nd integral of 1/r D of r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    raise RuntimeError('This operator should not be used. It is probably wrong!')
    ns = np.arange(0, nr+1)
    offsets = np.arange(-1,2)
    nzrow = 1

    # Generate 1st subdiagonal
    def d_1(n):
        return worland_norm_row(n,l,-1)*8.0*(l + n - 1.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate main diagonal
    def d0(n):
        return -worland_norm_row(n,l,0)*8.0*l/((l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -worland_norm_row(n,l,1)*2.0*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)/((l + n)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    ds = [d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)

def i4r_1d1_diags(nr, l):
    """Create operator for 4th integral of 1/r D of r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    raise RuntimeError('This operator should not be used. It is probably wrong!')
    ns = np.arange(0, nr+2)
    offsets = np.arange(-3,4)
    nzrow = 3

    # Generate 3rd subdiagonal
    def d_3(n):
        return worland_norm_row(n, l, -3)*32.0*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -worland_norm_row(n, l, -2)*96.0*l*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2*n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return worland_norm_row(n, l, -1)*24.0*(l + n - 1.0)*(4.0*l**2 - 4.0*l*n + 2.0*l - 4.0*n**2 + 4.0*n + 3.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate diagonal
    def d0(n):
        return -worland_norm_row(n, l, 0)*16.0*l*(2.0*l**2 - 12.0*l*n - 12.0*n**2 + 7.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -worland_norm_row(n, l, 1)*6.0*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)*(4.0*l**2 - 4.0*l*n - 2.0*l - 4.0*n**2 - 4.0*n + 3.0)/((l + n)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -worland_norm_row(n, l, 2)*6.0*l*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)/((l + n)*(l + n + 1.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -worland_norm_row(n, l, 3)*0.5*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)/((l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)

def i6r_1d1_diags(nr, m):
    """Create operator for 6th integral of 1/r D of r^m P_n^{-1/2,m-1/2}(2r^2-1)."""

    raise RuntimeError('This operator should not be used. It is probably wrong!')
    ns = np.arange(0, nr+3)
    offsets = np.arange(-5,6)
    nzrow = 5

    # Generate 5th subdiagonal
    def d_5(n):
        return worland_norm_row(n, m, -5)*128.0*(m + n - 5.0)*(m + n - 4.0)*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)/((m + 2.0*n - 10.0)*(m + 2.0*n - 9.0)*(m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return -worland_norm_row(n, m, -4)*640.0*m*(m + n - 4.0)*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)/((m + 2.0*n - 9.0)*(m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return worland_norm_row(n, m, -3)*160.0*(m + n - 3.0)*(m + n - 2.0)*(m + n - 1.0)*(8.0*m**2 - 4.0*m*n + 6.0*m - 4.0*n**2 + 12.0*n + 7.0)/((m + 2.0*n - 8.0)*(m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -worland_norm_row(n, m, -2)*640.0*m*(m + n - 2.0)*(m + n - 1.0)*(2.0*m**2 - 4.0*m*n + 4.0*m - 4.0*n**2 + 8.0*n + 9.0)/((m + 2.0*n - 7.0)*(m + 2.0*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return worland_norm_row(n, m, -1)*80.0*(m + n - 1.0)*(8.0*m**4 - 48.0*m**3*n + 24.0*m**3 - 32.0*m**2*n**2 + 32.0*m**2*n + 112.0*m**2 + 32.0*m*n**3 - 48.0*m*n**2 - 56.0*m*n + 36.0*m + 16.0*n**4 - 32.0*n**3 - 56.0*n**2 + 72.0*n + 45.0)/((m + 2*n - 6.0)*(m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0))

    # Generate diagonal
    def d0(n):
        return -worland_norm_row(n, m, 0)*16.0*m*(8.0*m**4 - 160.0*m**3*n + 80.0*m**2*n**2 + 260.0*m**2 + 480.0*m*n**3 - 920.0*m*n + 240.0*n**4 - 920.0*n**2 + 407.0)/((m + 2.0*n - 5.0)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -worland_norm_row(n, m, 1)*20.0*(2.0*n + 1.0)*(2.0*m + 2.0*n + 1.0)*(8.0*m**4 - 48.0*m**3*n - 24.0*m**3 - 32.0*m**2*n**2 - 32.0*m**2*n + 112.0*m**2 + 32.0*m*n**3 + 48.0*m*n**2 - 56.0*m*n - 36.0*m + 16.0*n**4 + 32.0*n**3 - 56.0*n**2 - 72.0*n + 45.0)/((m + n)*(m + 2.0*n - 4.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -worland_norm_row(n, m, 2)*40.0*m*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m**2 - 4.0*m*n - 4.0*m - 4.0*n**2 - 8.0*n + 9.0)/((m + n)*(m + n + 1.0)*(m + 2.0*n - 3.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -worland_norm_row(n, m, 3)*2.5*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(8.0*m**2 - 4.0*m*n - 6.0*m - 4.0*n**2 - 12.0*n + 7.0)/((m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + 2.0*n - 2.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -worland_norm_row(n, m, 4)*2.5*m*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(2.0*m + 2.0*n + 7.0)/((m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + n + 3.0)*(m + 2.0*n - 1.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0)*(m + 2.0*n + 9.0))

    # Generate 5rd superdiagonal
    def d5(n):
        return -worland_norm_row(n, m, 5)*0.125*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*n + 9.0)*(2.0*m + 2.0*n + 1.0)*(2.0*m + 2.0*n + 3.0)*(2.0*m + 2.0*n + 5.0)*(2.0*m + 2.0*n + 7.0)*(2.0*m + 2.0*n + 9.0)/((m + n)*(m + n + 1.0)*(m + n + 2.0)*(m + n + 3.0)*(m + n + 4.0)*(m + 2.0*n + 1.0)*(m + 2.0*n + 2.0)*(m + 2.0*n + 3.0)*(m + 2.0*n + 4.0)*(m + 2.0*n + 5.0)*(m + 2.0*n + 6.0)*(m + 2.0*n + 7.0)*(m + 2.0*n + 8.0)*(m + 2.0*n + 9.0)*(m + 2.0*n + 10.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)
    return (diags,offsets)
