"""Module provides the functions to generate the Boussinesq Rayleigh-Benard convection in a 3D box (streamfunction formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_3d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_3d import no_bc


class BoussinesqRBCBoxST(base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a 3D box (streamfunction formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "heating", "scale1d", "scale2d", "scale3d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("streamfunction",""), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("streamfunction",""), ("temperature","")]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("streamfunction",""), ("temperature","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]*res[1]*res[2]
        if self.use_galerkin:
            if field_row == ("streamfunction","") or field_row == ("temperature",""):
                shift_x = 2
                shift_y = 2
                shift_z = 2
            else:
                shift_x = 0
                shift_y = 0
                shift_z = 0

            gal_n = (res[0] - shift_x)*(res[1] - shift_y)*(res[2] - shift_z)

        else:
            gal_n = tau_n
            shift_x = 0
            shift_y = 0
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_x,shift_y,shift_z), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = False

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SINGLE

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            # No-slip/No-slip/No-slip, Fixed temperature/Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'y':{0:-40, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:40}, 'y':{0:40}, 'z':{0:40}, 'priority':'xz'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}

            # Stress-free/Stress-free/no-slip, Fixed flux/Fixed flux/Fixed temperature
            elif bcId == 4:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'y':{0:-41, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:41}, 'y':{0:41}, 'z':{0:40}, 'priority':'zsx'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'y':{0:21}, 'z':{0:20}, 'priority':'zsx'}

            # Stress-free/Stress-free/no-slip, Fixed flux/Fixed flux/Fixed temperature
            elif bcId == 6:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'y':{0:-41, 'rt':0}, 'z':{0:-40, 'rt':0}}

                else:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:41}, 'y':{0:41}, 'z':{0:41}, 'priority':'zsx'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['x']['rt'] = 4
                    bc['y']['rt'] = 4
                    bc['z']['rt'] = 4
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'y':{0:-40, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 4:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'y':{0:-41, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['x']['rt'] = 4
                    bc['y']['rt'] = 4
                    bc['z']['rt'] = 4
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction","") and field_col == field_row:
            mat = geo.i4j4k4(res[0], res[1], res[2], bc)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2k2(res[0], res[1], res[2], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']

        xscale = eq_params['scale1d']
        yscale = eq_params['scale2d']
        zscale = eq_params['scale3d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.i4j4k4lapl2(res[0], res[1], res[2], bc, xscale = xscale, yscale = yscale, zscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.i4j4k4(res[0], res[1], res[2], bc, -Ra)

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                mat = geo.i2j2k2laplh(res[0], res[1], res[2], bc, -1.0, xscale = xscale, yscale = yscale)

            elif field_col == ("temperature",""):
                mat = geo.i2j2k2lapl(res[0], res[1], res[2], bc, xscale = xscale, yscale = yscale, zscale = zscale)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']

        xscale = eq_params['scale1d']
        yscale = eq_params['scale2d']
        zscale = eq_params['scale3d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.i4j4k4lapl(res[0], res[1], res[2], bc, 1.0/Pr, xscale = xscale, yscale = yscale, zscale = zscale)

        elif field_row == ("temperature",""):
            mat = geo.i2j2k2(res[0], res[1], res[2], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
