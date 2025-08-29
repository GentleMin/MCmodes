"""Module provides functions to generate sparse operators in a cartesian box with two periodic directions."""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_boundary_1d as c1dbc


def zblk(nx, bc, location = 't'):
    """Create a block of zeros"""

    mat = spsp.coo_matrix((nx,nx))
    return c1dbc.constrain(mat,bc, location = location)

def z1(nx, bc, coeff = 1.0, cscale = 2.0, zr = 0, location = 't'):
    """Create operator for multiplication by x of T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-1,2,1)
    nzrow = -1

    cnst = coeff/cscale

    # Generate 1st subdiagonal
    def d_1(n):
        return (cnst/2.0)*np.ones(n.shape)

    # Generate diagonal
    def d0(n):
        return cnst*np.ones(n.shape)

    # Generate 1st superdiagonal
    def d1(n):
        return (cnst/2.0)*np.ones(n.shape)

    ds = [d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format = 'lil')
    if zr > 0 and location == 'b':
        mat[-zr:,:] = 0
    elif zr > 0 and location == 't':
        mat[0:zr,:] = 0
    mat = mat.tocoo()

    return c1dbc.constrain(mat, bc)

def d1(nx, bc, coeff = 1.0, cscale = 1.0, zr = 1):
    """Create operator for 1st derivative"""

    row = [coeff*cscale*2*j for j in range(0,nx)]
    mat = spsp.lil_matrix((nx,nx))
    for i in range(0,nx-1):
        mat[i,i+1:nx:2] = row[i+1:nx:2]
    mat[-zr:,:] = 0

    mat = mat.tocoo()
    return c1dbc.constrain(mat, bc, location = 'b')

def d2(nx, bc, coeff = 1.0, cscale = 1.0, zr = 2):
    """Create operator for 2nd derivative"""

    mat = spsp.lil_matrix((nx,nx))
    cnst = coeff*cscale**2
    for i in range(0,nx-2):
        mat[i,i+2:nx:2] = [cnst*j*(j**2 - i**2) for j in range(0,nx)][i+2:nx:2]
    mat[-zr:,:] = 0

    mat = mat.tocoo()
    return c1dbc.constrain(mat, bc, location = 'b')

def d4(nx, bc, coeff = 1.0, cscale = 1.0, zr = 4):
    """Create operator for 4th derivative"""

    mat_d2 = d2(nx + 4, c1dbc.no_bc(), cscale = cscale)
    mat = mat_d2*mat_d2
    mat = mat[0:-4, 0:-4]
    mat[-zr:,:] = 0
    
    mat = coeff*mat.tocoo()
    return c1dbc.constrain(mat, bc, location = 'b')

def lapl(nx, k, l, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for horizontal laplacian"""

    mat = d2(nx, bc, cscale = cscale) - k**2*sid(nx,2,bc) - l**2*sid(nx,2,bc)

    mat = coeff*mat.tocoo()
    return c1dbc.constrain(mat, bc, location = 'b')

def laplh(nx, k, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for horizontal laplacian"""

    mat = d2(nx, bc, cscale = cscale) - k**2*sid(nx,2,bc)

    mat = coeff*mat.tocoo()
    return c1dbc.constrain(mat, bc, location = 'b')

def lapl2h(nx, k, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for horizontal bilaplacian"""

    mat = d4(nx, bc, cscale = cscale) - 2.0*k**2*sid(nx,4,bc)*d2(nx,bc, cscale = cscale) + k**4*sid(nx,4,bc) 

    mat = coeff*mat.tocoo()
    return c1dbc.constrain(mat, bc, location = 'b')

def i1d1(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create a quasi identity block of order 1"""

    return qid(nx,1, bc, coeff*cscale)

def i2d2(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create a quasi identity block of order 2"""

    return qid(nx,2, bc, coeff*cscale**2)

def i2d4(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create a quasi identity block of order 2"""

    mat = i2d2(nx, c1dbc.no_bc(), cscale = cscale)*d2(nx, c1dbc.no_bc(), cscale = cscale)
    return c1dbc.constrain(mat, bc)

def i1(nx, bc, coeff = 1.0):
    """Create operator for 1st integral in x of T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-1,2,2)
    nzrow = 0

    # Generate 1st subdiagonal
    def d_1(n):
        return coeff/(2.0*n)

    # Generate 1st superdiagonal
    def d1(n):
        return -coeff/(2.0*n)

    ds = [d_1, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format = 'coo')
    return c1dbc.constrain(mat, bc)

def i1z1(nx, bc, coeff = 1.0, cscale = 2.0):
    """Create operator for 1st integral of multiplication by x in x of T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-2,3,1)
    nzrow = 0

    cnst = coeff/cscale

    # Generate 2nd subdiagonal
    def d_2(n):
        return cnst/(4.0*n)

    # Generate 1st subdiagonal
    def d_1(n):
        return cnst/(2.0*n)

    # Generate diagonal
    def d0(n):
        return cnst*np.zeros(n.shape)

    # Generate 1st subdiagonal
    def d1(n):
        return -cnst/(2.0*n)

    # Generate 1st superdiagonal
    def d2(n):
        return -cnst/(4.0*n)

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format = 'coo')
    return c1dbc.constrain(mat, bc)

def i2(nx, bc, coeff = 1.0):
    """Create operator for 2nd integral in x of T_n(x)"""
    
    ns = np.arange(0, nx, 1)
    offsets = np.arange(-2,3,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return coeff/(4.0*n*(n - 1.0))

    # Generate main diagonal
    def d0(n):
        return -coeff/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return coeff/(4.0*n*(n + 1.0))

    ds = [d_2, d0, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i2z1(nx, bc, coeff = 1.0, cscale = 2.0):
    """Create operator for 2nd integral in x of z of T_n(x)"""
    
    ns = np.arange(0, nx, 1)
    offsets = np.arange(-3,4,1)
    nzrow = 1

    cnst = coeff/cscale

    # Generate 3rd subdiagonal
    def d_3(n):
        return coeff/(8.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return coeff/(4.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -coeff/(8.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -coeff/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -coeff/(8.0*n*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return coeff/(4.0*n*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return coeff/(8.0*n*(n + 1.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i2x1(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 2nd integral in x of x T_n(x)"""
    
    ns = np.arange(0, nx, 1)
    offsets = np.arange(-3,4,2)
    nzrow = 1

    cnst = coeff*cscale

    # Generate 3rd subdiagonal
    def d_3(n):
        return cnst/(8.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -cnst/(8.0*n*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -cnst/(8.0*n*(n - 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return cnst/(8.0*n*(n + 1.0))

    ds = [d_3, d_1, d1, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i2d1(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 2nd integral in x of T_n(x)"""
    
    ns = np.arange(0, nx, 1)
    offsets = np.arange(-1,2,2)
    nzrow = 1

    cnst = coeff*cscale

    # Generate 1st subdiagonal
    def d_1(n):
        return cnst/(2.0*n)

    # Generate 1st superdiagonal
    def d1(n):
        return -cnst/(2.0*n)

    ds = [d_1, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i2lapl(nx, k, l, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 2nd integral in x of Laplacian T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-2,3,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return -coeff*(k**2 + l**2)/(4.0*n*(n - 1.0))

    # Generate main diagonal
    def d0(n):
        return coeff*(k**2 + l**2 + (2.0*n**2 - 2.0)*cscale**2)/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -coeff*(k**2 + l**2)/(4.0*n*(n + 1.0))

    ds = [d_2, d0, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i2laplh(nx, k, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 2nd integral in x of horizontal Laplacian T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-2,3,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return -coeff*k**2/(4.0*n*(n - 1.0))

    # Generate main diagonal
    def d0(n):
        return coeff*(k**2 + (2.0*n**2 - 2.0)*cscale**2)/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -coeff*k**2/(4.0*n*(n + 1.0))

    ds = [d_2, d0, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4(nx, bc, coeff = 1.0):
    """Create operator for 4th integral in x of T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-4,5,2)
    nzrow = 3
   
    # Generate 4th subdiagonal
    def d_4(n):
        return coeff/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -coeff/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)) 

    # Generate main diagonal
    def d0(n):
        return coeff*3.0/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -coeff/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0)) 

    # Generate 4th superdiagonal
    def d4(n):
        return coeff/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_4, d_2, d0, d2, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4d1(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 4th integral in x of D_x T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-3,4,2)
    nzrow = 3

    cnst = coeff*cscale

    # Generate 3rd subdiagonal
    def d_2(n):
        return cnst/(8.0*n*(n - 2.0)*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -cnst*3.0/(8.0*n*(n - 2.0)*(n + 1.0)) 

    # Generate 1st superdiagonal
    def d1(n):
        return cnst*3.0/(8.0*n*(n - 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -cnst/(8.0*n*(n + 1.0)*(n + 2.0)) 

    ds = [d_2, d_1, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4d2(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 4th integral in x of D_x^2 T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-2,3,2)
    nzrow = 3

    cnst = coeff*cscale**2

    # Generate 2nd subdiagonal
    def d_2(n):
        return cnst/(4.0*n*(n - 1.0))

    # Generate main diagonal
    def d0(n):
        return -cnst/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return cnst/(4.0*n*(n + 1.0))

    ds = [d_2, d0, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4d4(nx, bc, coeff = 1.0, cscale = 1.0):
    """Create a quasi identity block of order 4"""

    return qid(nx,4, bc, coeff*cscale**4)

def i4lapl(nx, k, l, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 4th integral in x of Laplacian T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-4,5,2)
    nzrow = 3
   
    # Generate 4th subdiagonal
    def d_4(n):
        return -coeff*(k**2 + l**2)/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return coeff*(k**2 + l**2 + (n**2 - 2.0*n - 3.0)*cscale**2)/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)) 

    # Generate main diagonal
    def d0(n):
        return -coeff*(3.0*k**2 + 3.0*l**2 + (4.0*n**2 - 16.0)*cscale**2)/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return coeff*(k**2 + l**2 + (n**2 + 2.0*n - 3.0)*cscale**2)/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -coeff*(k**2 + l**2)/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_4, d_2, d0, d2, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4laplh(nx, k, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 4th integral in x of horizontal Laplacian T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-4,5,2)
    nzrow = 3
   
    # Generate 4th subdiagonal
    def d_4(n):
        return -coeff*k**2/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return coeff*(k**2 + (n**2 - 2.0*n - 3.0)*cscale**2)/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)) 

    # Generate main diagonal
    def d0(n):
        return -coeff*(3.0*k**2 + (4.0*n**2 - 16.0)*cscale**2)/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return coeff*(k**2 + (n**2 + 2.0*n - 3.0)*cscale**2)/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -coeff*k**2/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_4, d_2, d0, d2, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4lapl2(nx, k, l, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 4th integral in x of Laplacian^2 T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-4,5,2)
    nzrow = 3

    # Generate 4th subdiagonal
    def d_4(n):
        return coeff*(k**2 + l**2)**2/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -coeff*(k**2 + l**2)*(k**2 + l**2 + (2.0*n**2 - 4.0*n - 6.0)*cscale**2)/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)) 

    # Generate main diagonal
    def d0(n):
        return coeff*(3.0*k**4 + 6.0*k**2*l**2 + 3.0*l**4 + (8.0*k**2*n**2 - 32.0*k**2 + 8.0*l**2*n**2 - 32.0*l**2)*cscale**2 + (8.0*n**4 - 40.0*n**2 + 32.0)*cscale**4)/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -coeff*(k**2 + l**2)*(k**2 + l**2 + (2.0*n**2 + 4.0*n - 6.0)*cscale**2)/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0)) 

    # Generate 4th superdiagonal
    def d4(n):
        return coeff*(k**2 + l**2)**2/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_4, d_2, d0, d2, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def i4lapl2h(nx, k, bc, coeff = 1.0, cscale = 1.0):
    """Create operator for 4th integral in x of horizontal Laplacian^2 T_n(x)"""

    ns = np.arange(0, nx, 1)
    offsets = np.arange(-4,5,2)
    nzrow = 3

    # Generate 4th subdiagonal
    def d_4(n):
        return coeff*k**4/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -coeff*k**2*(k**2 + (2.0*n**2 - 4.0*n - 6.0)*cscale**2)/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)) 

    # Generate main diagonal
    def d0(n):
        return coeff*(3.0*k**4 + (8.0*k**2*n**2 - 32.0*k**2)*cscale**2 + (8.0*n**4 - 40.0*n**2 + 32.0)*cscale**4)/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -coeff*k**2*(k**2 + (2.0*n**2 + 4.0*n - 6.0)*cscale**2)/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0)) 

    # Generate 4th superdiagonal
    def d4(n):
        return coeff*k**4/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_4, d_2, d0, d2, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format='coo')
    return c1dbc.constrain(mat, bc)

def qid(nx, q, bc, coeff = 1.0):
    """Create a quasi identity block of order q"""

    mat = spsp.coo_matrix((nx,nx))
    if coeff != 1.0:
        mat.data = coeff*np.ones((nx-q))
    else:
        mat.data = np.ones((nx-q))
    mat.row = np.arange(q,nx)
    mat.col = mat.row
    return c1dbc.constrain(mat, bc)

def sid(nx, s, bc, coeff = 1.0):
    """Create a identity block with last s rows zeroed"""

    mat = spsp.coo_matrix((nx,nx))
    if coeff != 1.0:
        mat.data = coeff*np.ones((nx-s))
    else:
        mat.data = np.ones((nx-s))
    mat.row = np.arange(0,nx-s)
    mat.col = mat.row
    return c1dbc.constrain(mat, bc, location = 'b')

def stencil(nx, bc, make_square):
    """Create a galerkin stencil matrix"""

    mat = qid(nx, 0, c1dbc.no_bc())

    if make_square:
        bc['rb'] = bc['rt']
    bc['rt'] = 0

    return c1dbc.constrain(mat, bc)

def avg(nx):
    """Compute the average of the expansion"""

    mat = spsp.lil_matrix((1,nx))
    mat[0,::2] = [2.0*(n/(n**2 - 1.0) - 1.0/(n - 1.0)) for n in np.arange(0,nx,2)]
    mat[0,0] = mat[0,0]/2.0

    return mat.tocoo()

def integral(nx, cscale = 1.0):
    """Compute the definite integral of the expansion"""

    mat = spsp.lil_matrix((1,nx))
    mat[0,::2] = [4.0*(n/(n**2 - 1.0) - 1.0/(n - 1.0)) for n in np.arange(0,nx,2)]
    mat[0,0] = mat[0,0]/2.0

    return mat.tocoo()

def surfaceFlux(nx, cscale = 1.0):
    """Compute the flux through boundary"""

    mat = c1dbc.constrain(spsp.lil_matrix((1, nx)), {0:12})
    mat = 2.0*cscale*mat

    return mat.tocoo()

def tau_mat(nx, tau, pad, bc, coeff = 1.0):
    """Create a quasi identity block of order q"""

    mat = spsp.coo_matrix((nx,nx))
    mat = c1dbc.constrain(mat, tau, pad_zeros = pad)
    return c1dbc.constrain(mat, bc)
