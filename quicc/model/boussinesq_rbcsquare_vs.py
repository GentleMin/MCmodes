"""Module provides the functions to generate the Boussinesq Rayleigh-Benard convection in a square cavity (2D) (vorticity-streamfunction formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_2d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_2d import no_bc


class BoussinesqRBCSquareVS(base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a square cavity (2D) (vorticity-streamfunction formulation)"""

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

        fields =  [("vorticity",""), ("streamfunction",""), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row in [("vorticity",""), ("streamfunction",""), ("temperature","")]:
            fields =  [("vorticity",""), ("streamfunction",""), ("temperature","")]

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
            if field_row in [("vorticity",""), ("temperature","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]*res[1]
        if self.use_galerkin:
            if field_row in [("streamfunction","")]:
                shift_x = 2
                shift_z = 2
            elif field_row in [("vorticity",""), ("temperature","")]:
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
                    if field_col == ("vorticity",""):
                        bc = {'x':{0:-2, 'rt':0}, 'z':{0:-2, 'rt':0}}
                    elif field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("vorticity","") and field_col == ("streamfunction",""):
                        #bc = {'x':{0:21}, 'z':{0:21}, 'priority':'x'}
                        bc = {'x':{0:40}, 'z':{0:40}, 'priority':'x'}
                    elif field_row == ("streamfunction","") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                        bc = no_bc()
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}

            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("vorticity","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:23}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("streamfunction","") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}

            # Stress-free/Stress-free, Fixed flux/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-42, 'rt':0}, 'z':{0:-42, 'rt':0}}

                else:
                    if field_row == ("vorticity","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:23}, 'z':{0:23}, 'priority':'z'}
                    elif field_row == ("streamfunction","") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'z'}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("vorticity",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("streamfunction",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
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
                        bc = {'x':{0:-42, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 2:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-42, 'rt':0}, 'z':{0:-42, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("vorticity",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("streamfunction",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
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
        if field_row == ("vorticity","") and field_col == field_row:
            mat = geo.i2j2(res[0], res[1], bc, restriction = restriction)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2(res[0], res[1], bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']

        xscale = eq_params['scale1d']
        zscale = eq_params['scale2d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("vorticity",""):
            if field_col == ("vorticity",""):
                mat = geo.i4j4lapl(res[0], res[1], 0, bc, xscale = xscale, zscale = zscale, restriction = restriction)
#                mat = mat.tolil()
#                mat[2, -2] = 1
#                mat[3, -1] = 1
#                mat[res[0]+2, -res[0]-2] = 1
#                mat[res[0]+3, -res[0]-1] = 1
#                #mat[-res[0]-2,:] = 0
#                #mat[-res[0]-1,:] = 0
#                #mat[-res[0]-2,-res[0]-2] = 1
#                #mat[-res[0]-1,-res[0]-1] = 1
##                mat[-2:,:] = 0
#                mat = mat.tocoo()

            elif field_col == ("streamfunction",""):
                mat = geo.zblk(res[0], res[1], 4, 4, bc)
#                mat = mat.tolil()
#                mat[2:3,:] = 0
#                mat[res[0]+2:res[0]+3,:] = 0
#                mat = mat.tocoo()

            elif field_col == ("temperature",""):
                mat = geo.i4j4d1(res[0], res[1], bc, Ra, xscale = xscale, restriction = restriction)
#                mat = mat.tolil()
                #mat[-res[0]-2,:] = 0
                #mat[-res[0]-1,:] = 0
#                mat[-2:,:] = 0
#                mat = mat.tocoo()

        elif field_row == ("streamfunction",""):
            s = 2
            bc['x']['rt'] = s
            bc['x']['cr'] = s
            bc['z']['rt'] = s
            bc['z']['cr'] = s
            if field_col == ("vorticity",""):
                mat = geo.i2j2(res[0]+s, res[1]+s, bc, restriction = restriction)
#                s = 2
#                bc['x']['rt'] = s
#                bc['x']['cr'] = s
#                bc['z']['rt'] = s
#                bc['z']['cr'] = s
#                tmp = geo.i2j2(res[0]+s, res[1]+s, bc, restriction = restriction)
#                tmp = tmp.tolil()
#                mat = mat.tolil()
#                mat[2, :] = tmp[-2*res[0]-2,:]
#                mat[3, :] = tmp[-2*res[0]-1,:]
#                mat[res[0]+2, :] = tmp[-3*res[0]-2,:]
#                mat[res[0]+3, :] = tmp[-3*res[0]-1,:]
#                mat = mat.tocoo()

            elif field_col == ("streamfunction",""):
                mat = geo.i2j2lapl(res[0]+s, res[1]+s, 0, bc, -1.0, xscale = xscale, zscale = zscale, restriction = restriction)
#                s = 2
#                bc['x']['rt'] = s
#                bc['x']['cr'] = s
#                bc['z']['rt'] = s
#                bc['z']['cr'] = s
#                tmp = geo.i2j2lapl(res[0]+s, res[1]+s, 0, bc, -1.0, xscale = xscale, zscale = zscale, restriction = restriction)
#                tmp = tmp.tolil()
#                mat = mat.tolil()
#                mat[2:3,:] = 0
#                mat[res[0]+2:res[0]+3,:] = 0
#                mat[2,:] = tmp[-2*res[0]-2,:]
#                mat[3,:] = tmp[-2*res[0]-1,:]
#                mat[res[0]+2, :] = tmp[-3*res[0]-2,:]
#                mat[res[0]+3, :] = tmp[-3*res[0]-1,:]
#                mat = mat.tocoo()

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0]+s, res[1]+s, 2, 2, bc)

        elif field_row == ("temperature",""):
            if field_col == ("vorticity",""):
                mat = geo.zblk(res[0], res[1], 2, 2, bc)

            elif field_col == ("streamfunction",""):
                if self.linearize or bcs["bcType"] == self.FIELD_TO_RHS:
                    if eq_params['heating'] == 0:
                        mat = geo.i2j2d1(res[0], res[1], bc, xscale = xscale, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[1], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2j2lapl(res[0], res[1], 0, bc, xscale = xscale, zscale = zscale, restriction = restriction)

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
        if field_row == ("vorticity",""):
            mat = geo.i4j4(res[0], res[1], bc, restriction = restriction)
#            mat = mat.tolil()
            #mat[-res[0]-2,:] = 0
            #mat[-res[0]-1,:] = 0
#            mat[-2:,:] = 0
#            mat = mat.tocoo()

        elif field_row == ("streamfunction",""):
            mat = geo.zblk(res[0], res[1], 2, 2, bc)

        elif field_row == ("temperature",""):
            mat = geo.i2j2(res[0], res[1], bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
