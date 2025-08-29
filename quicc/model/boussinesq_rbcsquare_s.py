"""Module provides the functions to generate the Boussinesq Rayleigh-Benard convection in a square cavity (2D) (streamfunction formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_2d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_2d import no_bc


class BoussinesqRBCSquareS(base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a square cavity (2D) (streamfunction formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "heating", "scale1d", "scale2d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("streamfunction",""), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row in [("streamfunction",""), ("temperature","")]:
            fields =  [("streamfunction",""), ("temperature","")]

        elif field_row in [("vorticityy","")]:
            fields = [("vorticityy","")]

        else:
            fields = []

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            if field_row == ("temperature",""):
                fields = [("streamfunction","")]
            else:
                fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("streamfunction",""), ("temperature","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            if field_row == ("vorticityy",""):
                fields = [("streamfunction","")]
            else:
                fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]*res[1]
        if self.use_galerkin:
            if field_row in [("streamfunction","")]:
                shift_x = 4
                shift_z = 4
            elif field_row in [("temperature","")]:
                shift_x = 2
                shift_z = 2
            else:
                shift_x = 0
                shift_z = 0

            gal_n = (res[0] - shift_x)*(res[1] - shift_z)

        else:
            gal_n = tau_n
            shift_x = 0
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_x,0,shift_z), 1)
        return block_info

    def stencil(self, res, eq_params, eigs, bcs, field_row, make_square):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return geo.stencil(res[0], res[1], bc, make_square)

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
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
            # No-slip/No-slip, Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("streamfunction","") and field_col == field_row:
                        bc = {'x':{0:40}, 'z':{0:40}, 'priority':'x'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}

            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("streamfunction","") and field_col == field_row:
                        bc = {'x':{0:41}, 'z':{0:40}, 'priority':'x'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}

            # Stress-free/Stress-free, Fixed flux/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:-41, 'rt':0}}

                else:
                    if field_row == ("streamfunction","") and field_col == field_row:
                        bc = {'x':{0:41}, 'z':{0:41}, 'priority':'z'}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['x']['rt'] = 4
                    bc['z']['rt'] = 4
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 1:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 2:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:-41, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['x']['rt'] = 4
                    bc['z']['rt'] = 4
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        xscale = eq_params['scale1d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("temperature","") and field_col == ("streamfunction",""):
            if eq_params['heating'] == 0:
                mat = geo.i2j2d1(res[0], res[1], bc, -1.0, xscale = xscale, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction","") and field_col == field_row:
            mat = geo.i4j4(res[0], res[1], bc, restriction = restriction)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2(res[0], res[1], bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nextstep_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit nextstep update"""

        xscale = eq_params['scale1d']
        zscale = eq_params['scale2d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("vorticityy","") and field_col == ("streamfunction",""):
            bc['x']['rt'] = 2
            bc['x']['cr'] = 2
            bc['z']['rt'] = 2
            bc['z']['cr'] = 2
            mat = geo.i2j2lapl(res[0]+2, res[1]+2, 0, bc, xscale = xscale, zscale = zscale, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']

        if True:
            # Diffusion time
            ns_diff = 1.0
            ns_temp = Ra
            t_diff = 1.0
        else:
            # Free fall
            ns_diff = (Pr/Ra)**0.5
            ns_temp = Pr
            t_diff = (Pr/Ra)**0.5

        xscale = eq_params['scale1d']
        zscale = eq_params['scale2d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.i4j4lapl2(res[0], res[1], 0, bc, ns_diff, xscale = xscale, zscale = zscale, restriction = restriction)

            elif field_col == ("temperature",""):
                mat = geo.i4j4d1(res[0], res[1], bc, ns_temp, xscale = xscale, restriction = restriction)

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                if self.linearize or bcs["bcType"] == self.FIELD_TO_RHS:
                    if eq_params['heating'] == 0:
                        mat = geo.i2j2d1(res[0], res[1], bc, xscale = xscale, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[1], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2j2lapl(res[0], res[1], 0, bc, t_diff, xscale = xscale, zscale = zscale, restriction = restriction)

        elif field_row == ("vorticityy","") and field_col == field_row:
            bc['x']['rt'] = 2
            bc['x']['cr'] = 2
            bc['z']['rt'] = 2
            bc['z']['cr'] = 2
            mat = geo.i2j2(res[0]+2, res[1]+2, bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']

        xscale = eq_params['scale1d']
        zscale = eq_params['scale2d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.i4j4lapl(res[0], res[1], 0, bc, xscale = xscale, zscale = zscale, restriction = restriction)

        elif field_row == ("temperature",""):
            mat = geo.i2j2(res[0], res[1], bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
