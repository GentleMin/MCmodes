"""Module provides the functions to generate the compressible rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (Toroidal/poloidal formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import sympy as sy
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as geo
import quicc.geometry.cartesian.cartesian_generic_1d as g1d
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class CompressibleRRBCPlaneConfig:
    """Class to setup the compressible rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (Toroidal/poloidal formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "ekman", "density_scale", "polytropic", "gamma", "scale1d", "scaling"]

    def automatic_parameters(self, eq_params):
        """Extend parameters with automatically computable values"""

        d = dict()
        dT = np.exp(eq_params['density_scale']/eq_params['polytropic'])-1.0
        d['Ha'] = eq_params['gamma']/((eq_params['polytropic'] + 1.0)*(eq_params['gamma']-1.0)*dT)
        d['Hs'] = 1.0/(dT - 1.0/d['Ha'])

        d['rayleigh'] = eq_params['rayleigh']/(1. + dT)**(2.0*eq_params['polytropic']-1.0)
        d['ekman'] = (eq_params['ekman']**(-2)/(1. + dT)**(2.0*eq_params['polytropic']))**(-0.5)

        ## Rescale Z direction with ekman number
        if eq_params['scaling'] == 1:
            d['aspect_ratio'] = eq_params["ekman"]**(1./3.)
        else:
            d['aspect_ratio'] = 1.0
        d['scale1d'] = eq_params["scale1d"]*d['aspect_ratio']

        return d

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocity", "entropy", "temperature"]

    def stencil(self, res, eq_params, eigs, bcs, field_row, make_square):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return geo.stencil(res[0], bc, make_square)

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = True

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.MODE

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

class CompressibleRRBCPlane(CompressibleRRBCPlaneConfig, base_model.BaseModel):
    """Class to setup the compressible rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (Toroidal/poloidal formulation)"""

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocity","x"), ("velocity","y"), ("velocity","z"), ("entropy",""), ("density",""), ("temperature",""), ("pressure","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature",""), ("entropy",""), ("density",""), ("pressure","")]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row in [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature","")]:
                shift_z = 2
            else:
                shift_z = 0

            gal_n = (res[0] - shift_z)

        else:
            gal_n = tau_n
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_z,0,0), 1)
        return block_info

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            m = eigs[0]

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            # No-slip / Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocity","y"):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocity","z"):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("entropy","") and field_col == ("temperature",""):
                        bc = {0:20}

            # Stress-free / Fixed flux
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("velocity","y"):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("velocity","z"):
                        bc = {0:-20, 'rt':0}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {0:21}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {0:21}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {0:20}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row in [("velocity","x"), ("velocity","y"), ("velocity","z")]:
                    bc['rt'] = 2
                elif field_row == ("entropy",""):
                    bc['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocity","x"):
                        bc = {0:-20}
                    elif field_col == ("velocity","y"):
                        bc = {0:-20}
                    elif field_col == ("velocity","z"):
                        bc = {0:-20}
                    elif field_col == ("temperature",""):
                        bc = {0:-20}

                elif bcId == 1:
                    if field_col == ("velocity","x"):
                        bc = {0:-21}
                    elif field_col == ("velocity","y"):
                        bc = {0:-21}
                    elif field_col == ("velocity","z"):
                        bc = {0:-20}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row in [("velocity","x"), ("velocity","y"), ("velocity","z")]:
                    bc['rt'] = 2
                elif field_row == ("entropy",""):
                    bc['rt'] = 2

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']
        Pr = eq_params['prandtl']
        E = eq_params['ekman']
        A = eq_params['aspect_ratio']
        Hs = eq_params['Hs']
        Ha = eq_params['Ha']
        poly = eq_params['polytropic']
        gamma = eq_params['gamma']
        zscale = eq_params['scale1d']

        kx = eigs[0]
        ky = eigs[1]

        x, Tbar, Rbar, Pbar, Sbar = self.make_bar(eq_params)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        # X velocity
        if field_row == ("velocity","x"):
            if field_col == ("velocity","x"):
                mat = geo.i2lapl(res[0], kx, ky, bc, 1.0, cscale = zscale)
                bc[0] = min(bc[0], 0)
                mat += geo.i2(res[0], bc, -(1./3.)*kx**2)

            elif field_col == ("velocity","y"):
                op = geo.i2(res[0], no_bc(), A**2/E)
                mat = g1d.mult_generic(op, res[0], 0, Rbar, x, bc)
                bc[0] = min(bc[0], 0)
                mat += geo.i2(res[0], bc, -(1./3.)*kx*ky)

            elif field_col == ("velocity","z"):
                mat = geo.i2d1(res[0], bc, (1./3.)*1j*kx, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("entropy",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("density",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("pressure",""):
                mat = geo.i2(res[0], bc, -1j*kx*Ra*Hs*A**2/Pr)

        # Y velocity
        elif field_row == ("velocity","y"):
            if field_col == ("velocity","x"):
                op = geo.i2(res[0], no_bc(), -A**2/E)
                mat = g1d.mult_generic(op, res[0], 0, Rbar, x, bc)
                bc[0] = min(bc[0], 0)
                mat += geo.i2(res[0], bc, -(1./3.)*kx*ky)

            elif field_col == ("velocity","y"):
                mat = geo.i2lapl(res[0], kx, ky, bc, 1.0, cscale = zscale)
                bc[0] = min(bc[0], 0)
                mat += geo.i2(res[0], bc, -(1./3.)*ky**2)

            elif field_col == ("velocity","z"):
                mat = geo.i2d1(res[0], bc, (1./3.)*1j*ky, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("entropy",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("density",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("pressure",""):
                mat = geo.i2(res[0], bc, -1j*ky*Ra*Hs*A**2/Pr)

        # Z velocity
        elif field_row == ("velocity","z"):
            if field_col == ("velocity","x"):
                mat = geo.i2d1(res[0], bc, (1./3.)*kx*1j, cscale = zscale)

            elif field_col == ("velocity","y"):
                mat = geo.i2d1(res[0], bc, (1./3.)*ky*1j, cscale = zscale)

            elif field_col == ("velocity","z"):
                mat = geo.i2lapl(res[0], kx, ky, bc, 1.0, cscale = zscale)
                bc[0] = min(bc[0], 0)
                mat += geo.i2d2(res[0], bc, (1./3.), cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("entropy",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("density",""):
                mat = geo.i2(res[0], bc, Ra*Hs*A**3/Pr)

            elif field_col == ("pressure",""):
                mat = geo.i2d1(res[0], bc, -Ra*Hs*A**2/Pr, cscale = zscale)

        # Density
        elif field_row == ("density",""):
            if field_col == ("velocity","x"):
                tbc = no_bc().copy()
                tbc['rt'] = bc.get('rt', 0) + 1
                tbc['cr'] = bc.get('cr', 0) + 1
                op = geo.i1(res[0]+1, no_bc(), -1j*kx)
                tbc = bc.copy()
                tbc['rt'] = bc.get('rt', 0) + 1
                tbc['cr'] = bc.get('cr', 0) + 1
                mat = g1d.mult_generic(op, res[0]+1, 0, Rbar, x, tbc)

            elif field_col == ("velocity","y"):
                tbc = no_bc().copy()
                tbc['rt'] = bc.get('rt', 0) + 1
                tbc['cr'] = bc.get('cr', 0) + 1
                op = geo.i1(res[0]+1, no_bc(), -1j*ky)
                tbc = bc.copy()
                tbc['rt'] = bc.get('rt', 0) + 1
                tbc['cr'] = bc.get('cr', 0) + 1
                mat = g1d.mult_generic(op, res[0]+1, 0, Rbar, x, tbc)

            elif field_col == ("velocity","z"):
                tbc = no_bc().copy()
                tbc['rt'] = bc.get('rt', 0) + 1
                tbc['cr'] = bc.get('cr', 0) + 1
                op = geo.i1d1(res[0]+1, no_bc(), -1.0, cscale = zscale)
                tbc = bc.copy()
                tbc['rt'] = bc.get('rt', 0) + 1
                tbc['cr'] = bc.get('cr', 0) + 1
                mat = g1d.mult_generic(op, res[0]+1, 0, Rbar, x, tbc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("entropy",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("density",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], bc)

        # Entropy
        elif field_row == ("entropy",""):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","z"):
                op = geo.i2(res[0], no_bc(), -1.0)
                mat = g1d.mult_generic(op, res[0], 0, Rbar*Tbar*sy.diff(2*Sbar, x), x, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2lapl(res[0], kx, ky, bc, 1.0/Pr, cscale = zscale)

            elif field_col == ("entropy",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("density",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], bc)

        # Pressure
        elif field_row == ("pressure",""):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                op = geo.qid(res[0], 0, no_bc(), -1.0)
                mat = g1d.mult_generic(op, res[0], 0, Pbar*Rbar, x, bc)

            elif field_col == ("entropy",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("density",""):
                op = geo.qid(res[0], 0, no_bc(), -1.0)
                mat = g1d.mult_generic(op, res[0], 0, Pbar*Tbar, x, bc)

            elif field_col == ("pressure",""):
                op = geo.qid(res[0], 0, no_bc())
                mat = g1d.mult_generic(op, res[0], 0, Rbar*Tbar, x, bc)

        # Entropy
        elif field_row == ("temperature",""):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("entropy",""):
                op = geo.qid(res[0], 0, no_bc())
                mat = g1d.mult_generic(op, res[0], 0, Pbar*Rbar, x, bc)

            elif field_col == ("density",""):
                op = geo.qid(res[0], 0, no_bc())
                mat = g1d.mult_generic(op, res[0], 0, Pbar, x, bc)

            elif field_col == ("pressure",""):
                op = geo.qid(res[0], 0, no_bc(), -1.0/gamma)
                mat = g1d.mult_generic(op, res[0], 0, Rbar, x, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        kx = eigs[0]
        ky = eigs[1]

        x, Tbar, Rbar, Pbar, Sbar = self.make_bar(eq_params)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        # X velocity
        if field_row == ("velocity","x"):
            op = geo.i2(res[0], no_bc(), 1.0)
            mat = g1d.mult_generic(op, res[0], 0, Rbar, x, bc)

        # Y velocity
        elif field_row == ("velocity","y"):
            op = geo.i2(res[0], no_bc(), 1.0)
            mat = g1d.mult_generic(op, res[0], 0, Rbar, x, bc)

        # Z velocity
        elif field_row == ("velocity","z"):
            op = geo.i2(res[0], no_bc(), 1.0)
            mat = g1d.mult_generic(op, res[0], 0, Rbar, x, bc)

        # Entropy
        elif field_row == ("entropy",""):
            op = geo.i2(res[0], no_bc(), 1.0)
            mat = g1d.mult_generic(op, res[0], 0, Rbar*Tbar, x, bc)

        # Density
        elif field_row == ("density",""):
            tbc = bc.copy()
            tbc['rt'] = bc.get('rt', 0) + 1
            tbc['cr'] = bc.get('cr', 0) + 1
            mat = geo.i1(res[0]+1, tbc, 1.0)

        # Pressure
        elif field_row == ("pressure",""):
            mat = geo.zblk(res[0], bc)

        # Pressure
        elif field_row == ("temperature",""):
            mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block of boundary operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        mat = geo.zblk(res[0], bc)
        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def make_bar(self, eq_params):
        """Make background states"""

        Hs = eq_params['Hs']
        Ha = eq_params['Ha']
        poly = eq_params['polytropic']
        gamma = eq_params['gamma']

        x = sy.Symbol('x')
        Tbar = (1.0 + (Hs**(-1) + Ha**(-1))*(x+1.)/2.)
        Rbar = (Tbar**poly)
        Pbar = (Ha*((gamma - 1.)/gamma)*Tbar**(poly+1))
        Sbar = (sy.log(Pbar**(1./gamma)/Rbar))

        return (x, Tbar, Rbar, Pbar, Sbar)

class CompressibleRRBCPlaneVisu(CompressibleRRBCPlaneConfig, base_model.BaseModel):
    """Class to setup the visualization options for TFF scheme """

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  []

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            if field_row in [("mean_temperature", ""),("fluct_temperature", "")]:
                fields = [("temperature","")]
            else:
                fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row in [("velocity","tor"), ("temperature","")]:
                shift_z = 2
            elif field_row in [("velocity","pol")]:
                shift_z = 4
            else:
                shift_z = 0

            gal_n = (res[0] - shift_z)

        else:
            gal_n = tau_n
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_z,0,0), 1)
        return block_info

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            raise RuntimeError("Equations are not setup properly!")

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            raise RuntimeError("Equations are not setup properly!")

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                raise RuntimeError("Equations are not setup properly!")

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        # Mean temperature
        if field_row == ("mean_temperature","") and field_col == ("temperature",""):
            if kx == 0 and ky == 0:
                mat = geo.qid(res[0], 0, bc)
            else:
                mat = geo.zblk(res[0], bc)
        elif field_row == ("fluct_temperature","") and field_col == ("temperature",""):
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0], bc)
            else:
                mat = geo.qid(res[0], 0, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
