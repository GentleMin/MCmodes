"""Module provides functions to generate sparse operators in a cartesian box."""

from __future__ import division
from __future__ import unicode_literals

import scipy.sparse as spsp

import quicc.geometry.cartesian.cartesian_1d as c1d
import quicc.geometry.cartesian.cartesian_2d as c2d
import quicc.geometry.cartesian.cartesian_boundary_3d as c3dbc
import quicc.base.utils as utils


def convert_bc(bc):
    """Convert boundary dictionary into x, y and z kronecker product boundaries"""

    if bc['x'][0] < 0:
        bcx = bc['x']
    else:
        bcx = c1d.c1dbc.no_bc()
        for key, val in bc['x'].items():
            if key != 0:
                bcx[key] = val

    if bc['y'][0] < 0:
        bcy = bc['y']
    else:
        bcy = c1d.c1dbc.no_bc()
        for key, val in bc['y'].items():
            if key != 0:
                bcy[key] = val

    if bc['z'][0] < 0:
        bcz = bc['z']
    else:
        bcz = c1d.c1dbc.no_bc()
        for key, val in bc['z'].items():
            if key != 0:
                bcz[key] = val

    return (bcx, bcy, bcz)

def d1(nx, ny, nz, bc, coeff = 1.0, sx = 1, sy = 0, sz = 0, xscale = 1.0, restriction = None):
    """Create operator for 1st X derivative of T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.sid(ny, sy, bcy), c1d.sid(nz, sz, bcz), c1d.d1(nx, bcx, cscale = xscale, zr = sx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, sx, sy, sz, bc, location = 'b', restriction = restriction)

def e1(nx, ny, nz, bc, coeff = 1.0, sx = 0, sy = 1, sz = 0, yscale = 1.0, restriction = None):
    """Create operator for 1st Y derivative of T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.d1(ny,bcy, cscale = yscale, zr = sy), c1d.sid(nz, sz, bcz), c1d.sid(nx, sx, bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, sx, sy, sz, bc, location = 'b', restriction = restriction)

def f1(nx, ny, nz, bc, coeff = 1.0, sx = 0, sy = 0, sz = 1, zscale = 1.0, restriction = None):
    """Create operator for 1st Z derivative of T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.sid(ny, sy, bcy), c1d.d1(nz, bcz, cscale = zscale, zr = sz), c1d.sid(nx, sx, bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, sx, sy, sz, bc, location = 'b', restriction = restriction)

def zblk(nx, ny, nz, qx, qy, qz, bc, location = 't', restriction = None):
    """Create a block of zeros"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = utils.restricted_kron_3d(c1d.zblk(ny,bcy), c1d.zblk(nz,bcz), c1d.zblk(nx,bcx), None)
    return c3dbc.constrain(mat, nx, ny, nz, qx, qy, qz, bc, location = location, restriction = restriction)

def i1j1k1d1(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, restriction = None):
    """Create operator for 1st integral in x,y,z of T'_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i1(ny,bcy), c1d.i1(nz, bcz), c1d.i1d1(nx, bcx, cscale = xscale), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 1, 1, 1, bc, restriction = restriction)

def i1j1k1e1(nx, ny, nz, bc, coeff = 1.0, yscale = 1.0, restriction = None):
    """Create operator for 1st integral in x,y,z of T_n(x)T'_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i1d1(ny,bcy, cscale = yscale), c1d.i1(nz, bcz), c1d.i1(nx, bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 1, 1, 1, bc, restriction = restriction)

def i1j1k1f1(nx, ny, nz, bc, coeff = 1.0, zscale = 1.0, restriction = None):
    """Create operator for 1st integral in x,y,z of T_n(x)T_n(y)T'_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i1(ny,bcy), c1d.i1d1(nz, bcz, cscale = zscale), c1d.i1(nx,bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 1, 1, 1, bc, restriction = restriction)

def i2j2k2d2e2f2(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, yscale = 1.0, zscale = 1.0, restriction = None):
    """Create a quasi identity block of order 2,2,2"""

    return qid(nx,ny,nz,2,2,2,bc,coeff*xscale**2*yscale**2*zscale**2, restriction = restriction)

def i2j2k2(nx, ny, nz, bc, coeff = 1.0, restriction = None):
    """Create operator for 2nd integral in x,y,z of T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i2(ny,bcy), c1d.i2(nz,bcz), c1d.i2(nx,bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 2, 2, 2, bc, restriction = restriction)

def i2j2k2d1(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, restriction = None):
    """Create operator for 2nd integral in x,y,z of T'_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i2(ny,bcy), c1d.i2(nz,bcz), c1d.i2d1(nx,bcx, cscale = xscale), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 2, 2, 2, bc, restriction = restriction)

def i2j2k2e1(nx, ny, nz, bc, coeff = 1.0, yscale = 1.0, restriction = None):
    """Create operator for 2nd integral in x,y,z of T_n(x)T'_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i2d1(ny,bcy,cscale = yscale), c1d.i2(nz,bcz), c1d.i2(nx,bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 2, 2, 2, bc, restriction = restriction)

def i2j2k2f1(nx, ny, nz, bc, coeff = 1.0, zscale = 1.0, restriction = None):
    """Create operator for 2nd integral in x,y,z of T_n(x)T_n(y)T'_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i2(ny,bcy), c1d.i2d1(nz,bcz, cscale = zscale), c1d.i2(nx,bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 2, 2, 2, bc, restriction = restriction)

def i2j2k2laplh(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, yscale = 1.0, restriction = None):
    """Create operator for 2nd integral in x,y,z of horizontal Laplacian T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = utils.restricted_kron_3d(c1d.i2(ny,bcy), c1d.i2(nz,bcz), c1d.i2d2(nx,bcx, cscale = xscale), restriction = restriction)
    bcx[0] = min(bcx[0], 0)
    bcy[0] = min(bcy[0], 0)
    bcz[0] = min(bcz[0], 0)
    mat = mat + utils.restricted_kron_3d(c1d.i2d2(ny,bcy, cscale = yscale), c1d.i2(nz,bcz), c1d.i2(nx,bcx), restriction = restriction)
    mat = coeff*mat
    return  c3dbc.constrain(mat, nx, ny, nz, 2, 2, 2, bc, restriction = restriction)

def i2j2k2lapl(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, yscale = 1.0, zscale = 1.0, restriction = None):
    """Create operator for 2nd integral in x,y,z of Laplacian T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = utils.restricted_kron_3d(c1d.i2(ny,bcy), c1d.i2(nz,bcz), c1d.i2d2(nx,bcx, cscale = xscale), restriction = restriction)
    bcx[0] = min(bcx[0], 0)
    bcy[0] = min(bcy[0], 0)
    bcz[0] = min(bcz[0], 0)
    mat = mat + utils.restricted_kron_3d(c1d.i2d2(ny,bcy, cscale = yscale), c1d.i2(nz,bcz), c1d.i2(nx,bcx), restriction = restriction)
    mat = mat + utils.restricted_kron_3d(c1d.i2(ny,bcy), c1d.i2d2(nz,bcz, cscale = zscale), c1d.i2(nx,bcx), restriction = restriction)
    mat = coeff*mat
    return  c3dbc.constrain(mat, nx, ny, nz, 2, 2, 2, bc, restriction = restriction)

def i4j4k4(nx, ny, nz, bc, coeff = 1.0, restriction = None):
    """Create operator for 4th integral in x,y,z of T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.i4(ny,bcy), c1d.i4(nz,bcz), c1d.i4(nx,bcx), restriction = restriction)
    return  c3dbc.constrain(mat, nx, ny, nz, 4, 4, 4, bc, restriction = restriction)

def i4j4k4lapl(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, yscale = 1.0, zscale = 1.0, restriction = None):
    """Create operator for 4th integral in x,y,z of Laplacian T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = utils.restricted_kron_3d(c1d.i4(ny,bcy), c1d.i4(nz,bcz), c1d.i4d2(nx,bcx, cscale = xscale), restriction = restriction)
    bcx[0] = min(bcx[0], 0)
    bcy[0] = min(bcy[0], 0)
    bcz[0] = min(bcz[0], 0)
    mat = mat + utils.restricted_kron_3d(c1d.i4d2(ny,bcy, cscale = yscale), c1d.i4(nz,bcz), c1d.i4(nx,bcx), restriction = restriction)
    mat = mat + utils.restricted_kron_3d(c1d.i4(ny,bcy), c1d.i4d2(nz,bcz, cscale = zscale), c1d.i4(nx,bcx), restriction = restriction)
    mat = coeff*mat
    return  c3dbc.constrain(mat, nx, ny, nz, 4, 4, 4, bc, restriction = restriction)

def i4j4k4lapl2(nx, ny, nz, bc, coeff = 1.0, xscale = 1.0, yscale = 1.0, zscale = 1.0, restriction = None):
    """Create operator for 4th integral in x,y,z of Laplacian^2 T_n(x)T_n(y)T_n(z)"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = utils.restricted_kron_3d(c1d.i4(ny,bcy), c1d.i4(nz,bcz), c1d.i4d4(nx,bcx, cscale = xscale), restriction = restriction)
    bcx[0] = min(bcx[0], 0)
    bcy[0] = min(bcy[0], 0)
    bcz[0] = min(bcz[0], 0)
    mat = mat + utils.restricted_kron_3d(c1d.i4d4(ny,bcy, cscale = yscale), c1d.i4(nz,bcz), c1d.i4(nx,bcx), restriction = restriction)
    mat = mat + utils.restricted_kron_3d(c1d.i4(ny,bcy), c1d.i4d4(nz,bcz, cscale = zscale), c1d.i4(nx,bcx), restriction = restriction)
    mat = mat + 2*utils.restricted_kron_3d(c1d.i4d2(ny,bcy, cscale = yscale), c1d.i4(nz,bcz),c1d.i4d2(nx,bcx, cscale = xscale), restriction = restriction)
    mat = mat + 2*utils.restricted_kron_3d(c1d.i4d2(ny,bcy, cscale = yscale), c1d.i4d2(nz,bcz,cscale = zscale), c1d.i4(nx,bcx), restriction = restriction)
    mat = mat + 2*utils.restricted_kron_3d(c1d.i4(ny,bcy), c1d.i4d2(nz,bcz,cscale = zscale), c1d.i4d2(nx,bcx, cscale = xscale), restriction = restriction)
    mat = coeff*mat
    return  c3dbc.constrain(mat, nx, ny, nz, 4, 4, 4, bc, restriction = restriction)

def qid(nx, ny, nz, qx, qy, qz, bc, coeff = 1.0, restriction = None):
    """Create a quasi identity block order qx,qy,qz"""

    bcx, bcy, bcz = convert_bc(bc)
    mat = coeff*utils.restricted_kron_3d(c1d.qid(ny,qy,bcy), c1d.qid(nz,qz,bcz), c1d.qid(nx,qx,bcx), restriction = restriction)
    return c3dbc.constrain(mat, nx, ny, nz, qx, qy, qz, bc, restriction = restriction)

def volumeAvg(nx, ny, nz):
    """Compute a volume average"""

    mat = c1d.avg(ny)*spsp.kron(c1d.qid(ny, 0, c1d.c1dbc.no_bc()), c2d.surfaceAvg(nx, nz))
    return mat

def avgFlux_z(nx, ny, nz, zscale = 1.0):
    """Compute average flux through Z surface"""

    mat = spsp.kron(c1d.avg(ny), spsp.kron(c1d.surfaceFlux(nz, cscale = zscale), c1d.avg(nx)))
    return mat
