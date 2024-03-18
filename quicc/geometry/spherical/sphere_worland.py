"""Module provides functions to generate sparse operators in a sphere with Worland polynomials in radius using a spherical harmonics expansion in the angular directions."""

from __future__ import division
from __future__ import unicode_literals

import scipy.sparse as spsp

import quicc.geometry.spherical.sphere_radius_worland as rad
import quicc.geometry.spherical.sphere_sh as sh
import quicc.geometry.spherical.sphere_boundary_worland as sphbc


def convert_bc(bc):
    """Convert boundary condition to be suitable for kronecker product boundaries"""

    if bc[0] < 0:
        bcr = bc
    else:
        bcr = rad.radbc.no_bc()
        for key, val in bc.items():
            if key != 0:
                bcr[key] = val

    return bcr

def sh_coeff(coeff):
    """Compute coefficients from spherical harmonic expansion"""

    if coeff == 'laplh':
        def fct(x):
            return x*(x + 1.0)
    else:
        def fct(x):
            return 1.0

    return fct

def fix_l_zero(nr, m, mat, bc, fix):
    """Fix problems with unused l = 0 modes"""

    if m > 0 or not fix:
        return mat
    elif fix == 'zero':
        return rad.zblk(nr, m, bc)
    elif fix == 'set':
        if bc[0] < 0:
            nn = nr - bc['rt']
        else:
            nn = nr
        return rad.qid(nn, m, 0, {0:0})
    else:
        raise RuntimeError("Unkown l=0 fix!")

def make_sh_loperator(op, nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Generic function to create a coupled l dependent spherical harmonics operator"""

    bcr = convert_bc(bc)
    shc = sh_coeff(with_sh_coeff)

    if restriction is None or m in restriction:
        bcr = sphbc.ldependent_bc(bcr, m)
        mat = coeff*shc(m)*op(nr, m, bcr)
        mat = fix_l_zero(nr, m, mat, bcr, l_zero_fix)
    else:
        mat = rad.zblk(nr, m, bcr)
    blocks = [mat]

    for l in range(m+1, maxnl):
        if restriction is None or l in restriction:
            bcr = sphbc.ldependent_bc(bcr, l)
            mat = coeff*shc(l)*op(nr, l, bcr)
        else:
            mat = rad.zblk(nr, l, bcr)
        blocks.append(mat)
    mat = spsp.block_diag(blocks, format = 'coo')

    return sphbc.constrain(mat, nr, maxnl, m, bc, l_zero_fix, restriction = restriction)

def make_sh_qoperator(opm, opp, nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create the coupled operator for the coriolis Q term"""

    # Only compute if there are at least 2 harmonic degrees
    if maxnl - m > 1:
        corm = sh.coriolism(maxnl, m)
        corp = sh.coriolisp(maxnl, m)

        if restriction is not None:
            corm = corm.tolil()
            corp = corp.tolil()
            res_cols = list(set(range(0,corp.shape[1])) - set([i - m for i in restriction]))
            corm[:,res_cols] = 0
            corp[:,res_cols] = 0

        corm = corm.tocsr()
        corp = corp.tocsr()

        bcr = convert_bc(bc)
        shc = sh_coeff(with_sh_coeff)

        assert(l_zero_fix == 'zero')
        bcr = sphbc.ldependent_bc(bcr, m)
        if m == 0:
            rmatm = spsp.lil_matrix((1,1))
            rmatp = spsp.lil_matrix((1,1))
        else:
            rmatm = opm(nr, m, bcr)
            rmatp = opp(nr, m, bcr)
        rmatm = fix_l_zero(nr, m, rmatm, bcr, l_zero_fix)
        rmatp = fix_l_zero(nr, m, rmatp, bcr, l_zero_fix)
        mat = coeff*shc(m)*spsp.kron(corm[0,:], rmatm, format = 'coo') + coeff*shc(m)*spsp.kron(corp[0,:],rmatp, format = 'coo')
        rows = [mat]
        for ir,l in enumerate(range(m+1, maxnl)):
            bcr = sphbc.ldependent_bc(bcr, l-1)
            rmatm = opm(nr, l, bcr)
            bcr = sphbc.ldependent_bc(bcr, l+1)
            rmatp = opp(nr, l, bcr)
            mat = coeff*shc(l)*spsp.kron(corm[ir+1,:], rmatm, format = 'coo') + coeff*shc(l)*spsp.kron(corp[ir+1,:],rmatp, format = 'coo')
            rows.append(mat)
        mat = spsp.vstack(rows)
    else:
        mat = rad.zblk(nr, m, bc)

    return sphbc.constrain(mat, nr, maxnl, m, bc, l_zero_fix, restriction = restriction)

def zblk(nr, maxnl, m, bc, l_zero_fix = False, restriction = None):
    """Create a block of zeros"""

    return make_sh_loperator(rad.zblk, nr, maxnl, m, bc, 0.0, with_sh_coeff = None, l_zero_fix = l_zero_fix, restriction = restriction)

def i1(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i1 radial operator kronecker with an identity"""

    return make_sh_loperator(rad.i1, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i2(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2 radial operator kronecker with an identity"""

    return make_sh_loperator(rad.i2, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i2lapl(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2lapl radial operator kronecker with an identity"""

    return make_sh_loperator(rad.i2lapl, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i4(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i4 radial operator kronecker with an identity"""

    return make_sh_loperator(rad.i4, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i4lapl(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i4lapl radial operator kronecker with an identity"""

    return make_sh_loperator(rad.i4lapl, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i4lapl2(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i4lapl2 radial operator kronecker with an identity"""

    return make_sh_loperator(rad.i4lapl2, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i1coriolis(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i1 radial operator kronecker with coriolis Q term"""

    return make_sh_qoperator(rad.i1qm, rad.i1qp, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i2coriolis(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2 radial operator kronecker with coriolis Q term"""

    return make_sh_qoperator(rad.i2qm, rad.i2qp, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def i4coriolis(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i4 radial operator kronecker with coriolis Q term"""

    return make_sh_qoperator(rad.i4qm, rad.i4qp, nr, maxnl, m, bc, coeff, with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)

def qid(nr, maxnl, m, qr, bc, coeff = 1.0):
    """Create a quasi identity block order qr in r"""

    bcr = convert_bc(bc)

    mat = coeff*rad.qid(nr, m, bcr)
    for l in range(m+1, maxnl):
        mat = spsp.block_diag((mat,coeff*rad.qid(nr, l, qr, bcr)), format = 'coo')

    return sphbc.constrain(mat, nr, maxnl, m, bc)

def stencil(nr, maxnl, m, bc, make_square):
    """Create a galerkin stencil matrix"""

    bcr = convert_bc(bc)

    bcr = sphbc.ldependent_bc(bcr, m)
    mat = rad.stencil(nr, m, bcr, make_square)

    for l in range(m+1, maxnl):
        bcr = sphbc.ldependent_bc(bcr, l)
        mat = spsp.block_diag((mat,rad.stencil(nr, l, bcr, make_square)))

    return mat
