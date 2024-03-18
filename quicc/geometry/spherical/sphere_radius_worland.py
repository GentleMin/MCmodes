"""Module provides functions to generate sparse operators for the radial direction in a sphere with Worland expansion."""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.sphere_radius_boundary_worland as radbc
import quicc.geometry.worland.worland_basis as wb


def zblk(nr, l, bc):
    """Create a block of zeros"""

    # Copy BC dict as we modify it!
    bc = dict(bc)

    mat = spsp.coo_matrix((nr,nr))
    return radbc.constrain(mat,l,bc)

def r2(nr, l, bc, coeff = 1.0, zr = 0):
    """Create operator for 1st integral r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-1,2)
    nzrow = -1

    # Generate 1st subdiagonal
    def d_1(n):
        if l == 0: # Continuity
            val = wb.worland_norm_row(n,l,-1)*n/(2.0*(l + 2.0*n - 1.0))
            val[0] = wb.worland_norm_row(n[0:1],l,-1)*1.0/(l + 1.0)
        else:
            val = wb.worland_norm_row(n,l,-1)*n*(l + n - 1.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))
        return val

    # Generate main diagonal
    def d0(n):
        if l == 1:
            val = wb.worland_norm_row(n,l,0)*(1.0 + n)/(2.0*(n + 1.0))
        else:
            val = wb.worland_norm_row(n,l,0)*(2.0*l**2 + 4.0*l*n - l + 4.0*n**2 - 1.0)/(2*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))
        if l == 0 or l == 1: # Continuity
            val[0] = wb.worland_norm_row(n[0:1],l,0)*(2.0*l + 1.0)/(2.0*(l + 1.0))
        return val

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n,l,1)*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)/(4.0*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    ds = [d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    if zr > 0:
        mat = mat.tolil()
        mat[-zr:,:] = 0
        mat = mat.tocoo()
    return radbc.constrain(mat, l, bc)

def i1(nr, l, bc, coeff = 1.0):
    """Create operator for 1st integral r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    diags,offsets = wb.i1_diags(nr+1, l)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i1qm(nr, l, bc, coeff = 1.0):
    """Create operator for 1st integral of Q r^{l-1} P_n^{-1/2,l-3/2}(2r^2 -1)."""

    ns = np.arange(0, nr+1)
    offsets = np.arange(0,2)
    nzrow = 0

    # Generate main diagonal
    def d0(n):
        return -wb.worland_norm_row(n,l,0,-1)*4.0*(l + n - 1.0)/(l + 2.0*n - 1.0)

    # Generate 1st superdiagonal
    def d1(n):
        return -wb.worland_norm_row(n,l,1,-1)*2.0*(2.0*n + 1.0)/(l + 2.0*n + 1.0)

    ds = [d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i1qp(nr, l, bc, coeff = 1.0):
    """Create operator for 1st integral of Q r^{l+1} P_n^{-1/2,l+1/2}(2r^2 -1)."""

    ns = np.arange(0, nr+1)
    offsets = np.arange(-1,1)
    nzrow = 0

    # Generate 1st subdiagonal
    def d_1(n):
        return wb.worland_norm_row(n,l,-1,1)*2.0*(2.0*l + 2.0*n + 1.0)/(l + 2.0*n - 1.0)

    # Generate main diagonal
    def d0(n):
        return wb.worland_norm_row(n,l,0,1)*(2.0*n - 1.0)*(2.0*l + 2.0*n + 1.0)/((l + n)*(l + 2.0*n + 1.0))

    ds = [d_1, d0]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i2(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral r^l P_n^{-1/2,l-1/2}(2r^2 -1)."""

    diags,offsets = wb.i2_diags(nr, l)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i2lapl(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral of Laplacian r^l P_n^{-1/2,l-1/2}(2r^2 -1)."""

    ns = np.arange(0, nr+1)
    offsets = np.arange(-1,2)
    nzrow = 1

    # Generate 1st subdiagonal
    def d_1(n):
        return wb.worland_norm_row(n,l,-1)*8.0*(l + n - 1.0)*(2.0*l + 2.0*n - 1.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate main diagonal
    def d0(n):
        return wb.worland_norm_row(n,l,0)*8.0*(4.0*l*n + 4.0*n**2 - 1.0)/((l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n,l,1)*2.0*(2.0*n + 1.0)**2*(2.0*l + 2.0*n + 1.0)/((l + n)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    ds = [d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i2qm(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral of Q r^{l-1} P_n^{-1/2,l-3/2}(2r^2 -1)."""

    ns = np.arange(0, nr+1)
    offsets = np.arange(-1,3)
    nzrow = 1

    # Generate 1st subdiagonal
    def d_1(n):
        return -wb.worland_norm_row(n,l,-1,-1)*8.0*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate main diagonal
    def d0(n):
        return wb.worland_norm_row(n,l,0,-1)*4.0*(l + n - 1.0)*(2.0*l - 2.0*n - 1.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n,l,1,-1)*2.0*(2.0*n + 1.0)*(4.0*l + 2.0*n - 1.0)/((l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return wb.worland_norm_row(n,l,2,-1)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)/((l + n)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    ds = [d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i2qp(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral of Q r^{l+1} P_n^{-1/2,l+1/2}(2r^2 -1)."""

    ns = np.arange(0, nr+1)
    offsets = np.arange(-2,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return wb.worland_norm_row(n,l,-2,1)*4.0*(l + n - 1.0)*(2.0*l + 2.0*n - 1.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -wb.worland_norm_row(n,l,-1,1)*2.0*(4.0*l**2 + 4.0*l - 4.0*n**2 + 4.0*n + 3.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -wb.worland_norm_row(n,l,0,1)*(2.0*l + 2.0*n + 1.0)*(8.0*l*n + 4.0*n**2 + 4.0*n - 3.0)/((l + n)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -wb.worland_norm_row(n,l,1,1)*(2.0*n + 1.0)**2*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)/(2.0*(l + n)*(l + n + 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    ds = [d_2, d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)

def i4(nr, l, bc, coeff = 1.0):
    """Create operator for 4th integral r^l P_n^{-1/2,l-1/2}(2r^2-1)."""

    diags,offsets = wb.i4_diags(nr, l)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 2)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 2)
    return radbc.constrain(mat, l, bc)

def i4lapl(nr, l, bc, coeff = 1.0):
    """Create operator for 4th integral of Laplacian r^l P_n^{-1/2,l-1/2}(2r^2 -1)."""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-3,4)
    nzrow = 3

    # Generate 3rd subdiagonal
    def d_3(n):
        return wb.worland_norm_row(n,l,-3)*32.0*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)*(2.0*l + 2.0*n - 5.0)/((l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -wb.worland_norm_row(n,l,-2)*32.0*(l + n - 2.0)*(l + n - 1.0)*(4.0*l**2 - 6.0*l - 4.0*n**2 + 8.0*n + 5.0)/((l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return wb.worland_norm_row(n,l,-1)*8.0*(l + n - 1.0)*(8.0*l**3 - 40.0*l**2*n - 4.0*l**2 - 56.0*l*n**2 + 80.0*l*n + 62.0*l - 8.0*n**3 + 36.0*n**2 - 22.0*n - 21.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate main diagonal
    def d0(n):
        return wb.worland_norm_row(n,l,0)*16.0*(16.0*l**3*n + 4.0*l**3 - 24.0*l**2*n - 24.0*l**2 - 32.0*l*n**3 - 24.0*l*n**2 + 40.0*l*n + 14.0*l - 16.0*n**4 + 40.0*n**2 - 9.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n,l,1)*2.0*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)*(48.0*l**2*n + 48.0*l**2 + 32.0*l*n**2 + 8.0*l*n - 84.0*l - 8.0*n**3 - 36.0*n**2 - 22.0*n + 21.0)/((l + n)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return wb.worland_norm_row(n,l,2)*2.0*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(8.0*l*n + 14.0*l + 4.0*n**2 + 8.0*n - 5.0)/((l + n)*(l + n + 1.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return wb.worland_norm_row(n,l,3)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)**2*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)/(2.0*(l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 2)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 2)
    return radbc.constrain(mat, l, bc)

def i4lapl2(nr, l, bc, coeff = 1.0):
    """Create operator for 4th integral bilaplacian r^l P_n^{-1/2, l-1/2}(2r^2 - 1)."""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-2,3)
    nzrow = 3

    # Generate 2nd subdiagonal
    def d_2(n):
        return wb.worland_norm_row(n,l,-2)*64.0*(l + n - 2.0)*(l + n - 1.0)*(2.0*l + 2.0*n - 5.0)*(2.0*l + 2.0*n - 3.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return wb.worland_norm_row(n,l,-1)*128.0*(l + n - 1.0)*(2.0*l + 2.0*n - 3.0)*(4.0*l*n + 4.0*n**2 - 4.0*n - 3.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate diagonal
    def d0(n):
        return wb.worland_norm_row(n,l,0)*96.0*(2.0*n + 1.0)*(2.0*l + 2.0*n - 1.0)*(4.0*l*n + 2.0*l + 4.0*n**2 - 5.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n,l,1)*32.0*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(4.0*l*n + 4.0*l + 4.0*n**2 + 4.0*n - 3.0)/((l + n)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return wb.worland_norm_row(n,l,2)*4.0*(2.0*n + 1.0)*(2.0*n + 3.0)**2*(2.0*n + 5.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)/((l + n)*(l + n + 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 2)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 2)
    return radbc.constrain(mat, l, bc)

def i4qm(nr, l, bc, coeff = 1.0):
    """Create operator for 4th integral of Q r^{l-1} P_n^{-1/2, l-3/2}(2r^2 - 1)."""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-3,5)
    nzrow = 3

    # Generate 3rd subdiagonal
    def d_3(n):
        return - wb.worland_norm_row(n,l,-3,-1)*32.0*(l + n - 4.0)*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)/((l + 2.0*n - 7.0)*(l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return  wb.worland_norm_row(n,l,-2,-1)*16.0*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)*(6.0*l - 2.0*n - 1.0)/((l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return - wb.worland_norm_row(n,l,-1,-1)*24.0*(l + n - 2.0)*(l + n - 1.0)*(4.0*l**2 - 8.0*l*n - 4.0*n**2 + 8.0*n + 5.0)/((l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate diagonal
    def d0(n):
        return  wb.worland_norm_row(n,l,0,-1)*4.0*(l + n - 1.0)*(8.0*l**3 - 72.0*l**2*n - 12.0*l**2 - 24.0*l*n**2 + 72.0*l*n + 58.0*l + 24.0*n**3 + 12.0*n**2 - 54.0*n - 27.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return  wb.worland_norm_row(n,l,1,-1)*2.0*(2.0*n + 1.0)*(32.0*l**3 - 48.0*l**2*n - 72.0*l**2 - 96.0*l*n**2 - 48.0*l*n + 112.0*l - 24.0*n**3 + 12.0*n**2 + 54.0*n - 27.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return  wb.worland_norm_row(n,l,2,-1)*3.0*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(8.0*l**2 - 8.0*l - 4.0*n**2 - 8.0*n + 5.0)/((l + n)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return  wb.worland_norm_row(n,l,3,-1)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(8.0*l + 2.0*n - 1.0)/(2.0*(l + n)*(l + n + 1.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0))

    # Generate 4th superdiagonal
    def d4(n):
        return  wb.worland_norm_row(n,l,4,-1)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)*(2.0*n + 7.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)/(4.0*(l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0)*(l + 2.0*n + 7.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 2)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 2)
    return radbc.constrain(mat, l, bc)

def i4qp(nr, l, bc, coeff = 1.0):
    """Create operator for 4th integral of Q r^{l+1} P_n^{-1/2, l+1/2}(2r^2 - 1)."""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-4,4)
    nzrow = 3

    # Generate 4th subdiagonal
    def d_4(n):
        return wb.worland_norm_row(n,l,-4,1)*16.0*(l + n - 3.0)*(l + n - 2.0)*(l + n - 1.0)*(2.0*l + 2.0*n - 5.0)/((l + 2.0*n - 7.0)*(l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -wb.worland_norm_row(n,l,-3,1)*8.0*(l + n - 2.0)*(l + n - 1.0)*(12.0*l**2 + 8.0*l*n - 16.0*l - 4.0*n**2 + 12.0*n + 7.0)/((l + 2.0*n - 6.0)*(l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return wb.worland_norm_row(n,l,-2,1)*12.0*(l + n - 1.0)*(8.0*l**3 - 8.0*l**2*n + 4.0*l**2 - 24.0*l*n**2 + 40.0*l*n + 26.0*l - 8.0*n**3 + 20.0*n**2 + 2.0*n - 5.0)/((l + 2.0*n - 5.0)*(l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -wb.worland_norm_row(n,l,-1,1)*2.0*(16.0*l**4 - 128.0*l**3*n + 32.0*l**3 - 192.0*l**2*n**2 + 96.0*l**2*n + 272.0*l**2 + 32.0*l*n + 112.0*l + 48.0*n**4 - 48.0*n**3 - 144.0*n**2 + 108.0*n + 81.0)/((l + 2.0*n - 4.0)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    # Generate diagonal
    def d0(n):
        return -wb.worland_norm_row(n,l,0,1)*(2.0*l + 2.0*n + 1.0)*(64.0*l**3*n + 16.0*l**3 - 96.0*l**2*n**2 - 48.0*l**2*n - 96.0*l**2 - 192.0*l*n**3 - 144.0*l*n**2 + 320.0*l*n - 4.0*l - 48.0*n**4 - 48.0*n**3 + 144.0*n**2 + 108.0*n - 81.0)/((l + n)*(l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -wb.worland_norm_row(n,l,1,1)*3.0*(2.0*n + 1.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(16.0*l**2*n + 16.0*l**2 - 24.0*l - 8.0*n**3 - 20.0*n**2 + 2.0*n + 5.0)/(2.0*(l + n)*(l + n + 1.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -wb.worland_norm_row(n,l,2,1)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)*(16.0*l*n + 28.0*l + 4.0*n**2 + 12.0*n - 7.0)/(4.0*(l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -wb.worland_norm_row(n,l,3,1)*(2.0*n + 1.0)*(2.0*n + 3.0)*(2.0*n + 5.0)**2*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)*(2.0*l + 2.0*n + 5.0)*(2.0*l + 2.0*n + 7.0)/(8.0*(l + n)*(l + n + 1.0)*(l + n + 2.0)*(l + n + 3.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0)*(l + 2.0*n + 4.0)*(l + 2.0*n + 5.0)*(l + 2.0*n + 6.0)*(l + 2.0*n + 7.0))

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 2)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 2)
    return radbc.constrain(mat, l, bc)

def qid(nr, l, q, bc, coeff = 1.0):
    """Create a quasi identity block of order q"""
    
    mat = spsp.coo_matrix((nr,nr))
    if coeff != 1.0:
        mat.data = coeff*np.ones((nr-q))
    else:
        mat.data = np.ones((nr-q))
    mat.row = np.arange(q,nr)
    mat.col = mat.row
    return radbc.constrain(mat, l, bc)

def stencil(nr, l, bc, make_square):
    """Create a galerkin stencil matrix"""

    mat = qid(nr, l, 0, radbc.no_bc())

    if make_square:
        bc['rb'] = bc['rt']
    bc['rt'] = 0

    return radbc.constrain(mat, l, bc)
