"""Module provides the functions to generate the Boussinesq rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (rescaled formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class BoussinesqRescaledRRBCPlane(base_model.BaseModel):
    """Class to setup the Boussinesq rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (rescaled formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "ekman", "heating", "scale1d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature", "pressure"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("streamfunction",""), ("velocityz",""), ("temperature",""), ("pressure","")]
        #fields =  [("streamfunction",""), ("velocityz",""), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("streamfunction",""), ("velocityz",""), ("temperature",""), ("pressure","")]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("streamfunction",""), ("velocityz",""), ("temperature",""), ("pressure","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            if field_row == ("vorticityz",""):
                fields = [("streamfunction","")]
            else:
                fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row in [("streamfunction",""), ("velocityz",""), ("temperature",""), ("pressure","")]:
                shift_z = 2
            else:
                shift_z = 0

            gal_n = (res[0] - shift_z)

        else:
            gal_n = tau_n
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_z,0,0), 1)
        return block_info

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
                    if field_col == ("streamfunction",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocityz",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("pressure",""):
                        bc = {0:-20, 'rt':0}

                else:
                    if field_row == ("streamfunction","") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("velocityz","") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {0:20}
                    #elif field_row == ("pressure","") and field_col == field_row:
                    #    bc = {0:20}
                    elif field_row == ("pressure","") and field_col == ("velocityz",""):
                        bc = {0:23}

            # Stress-free / Fixed flux
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("velocityz",""):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("pressure",""):
                        bc = {0:-21, 'rt':0}

                else:
                    if field_row == ("streamfunction","") and field_col == field_row:
                        bc = {0:21}
                    elif field_row == ("velocityz","") and field_col == field_row:
                        bc = {0:21}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {0:21}
                    #elif field_row == ("pressure","") and field_col == field_row:
                    #    bc = {0:21}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['rt'] = 2
                elif field_row == ("velocityz",""):
                    bc['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['rt'] = 2
                elif field_row == ("pressure",""):
                    bc['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("streamfunction",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocityz",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("pressure",""):
                        bc = {0:-20, 'rt':0}

                elif bcId == 1:
                    if field_col == ("streamfunction",""):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("velocityz",""):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("pressure",""):
                        bc = {0:-21, 'rt':0}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['rt'] = 2
                elif field_row == ("velocityz",""):
                    bc['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['rt'] = 2
                elif field_row == ("pressure",""):
                    bc['rt'] = 2

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction","") and field_col == field_row:
            mat = geo.i2(res[0], bc)

        elif field_row == ("velocityz","") and field_col == field_row:
            mat = geo.i2(res[0], bc)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2(res[0], bc)

        elif field_row == ("pressure","") and field_col == field_row:
            mat = geo.i2(res[0], bc)

        elif field_row == ("dz_meantemperature","") and field_col == field_row:
            if eigs[0] == 0 and eigs[1] == 0:
                mat = (geo.qid(res[0],0, bc) - spsp.eye(res[0], 1)*geo.avg(res[0]))
            else:
                mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']
        Pr = eq_params['prandtl']
        E = eq_params['ekman']
        Ro = E**(1./3.)
        zscale = eq_params['scale1d']

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.i2(res[0], bc, (kx**2 + ky**2)**2)
                bc[0] = min(bc[0], 0)
                mat += geo.i2d2(res[0], bc, -Ro**2*(kx**2 + ky**2), cscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.i2d1(res[0], bc, 1.0, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], bc)

        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.i2d1(res[0], bc, -1.0, cscale = zscale)
                #mat = geo.i2d1(res[0], bc, -(1.0 - Ro**2), cscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.i2(res[0], bc, -(kx**2 + ky**2))
                bc[0] = min(bc[0], 0)
                mat += geo.i2d2(res[0], bc, Ro**2, cscale = zscale)

            elif field_col == ("temperature",""):
                if kx == 0 and ky == 0:
                    mat = geo.zblk(res[0], bc)
                else:
                    mat = geo.i2(res[0], bc, (Ra/Pr))

            elif field_col == ("pressure",""):
                mat = geo.i2d1(res[0], bc, -Ro**2, cscale = zscale)

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocityz",""):
                if self.linearize:
                    mat = geo.i2(res[0], bc, 1.0)
                else:
                    mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.i2(res[0], bc, -(kx**2 + ky**2)/Pr)
                bc[0] = min(bc[0], 0)
                mat += geo.i2d2(res[0], bc, Ro**2/Pr, cscale = zscale)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], bc)

        elif field_row == ("pressure",""):
            if field_col == ("streamfunction",""):
                mat = geo.i2d2(res[0], bc, 1.0, cscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.i2d1(res[0], bc, -Ra/Pr, cscale = zscale)

            elif field_col == ("pressure",""):
                mat = geo.i2(res[0], bc, -(kx**2 + ky**2))
                bc[0] = min(bc[0], 0)
                mat += geo.i2d2(res[0], bc, Ro**2, cscale = zscale)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.i2(res[0], bc, -(kx**2 + ky**2))

        elif field_row == ("velocityz",""):
            mat = geo.i2(res[0], bc, 1.0)

        elif field_row == ("temperature",""):
            mat = geo.i2(res[0], bc, 1.0)

        elif field_row == ("pressure",""):
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
