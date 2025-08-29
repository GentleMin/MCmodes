"""Module provides the functions to generate the Boussinesq Beta 3DQG model"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_2d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_2d import no_bc


class BoussinesqBeta3DQG(base_model.BaseModel):
    """Class to setup the Boussinesq Beta 3DQG model"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "gamma", "chi", "scale1d", "scale3d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature", "vorticityz"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields = [("streamfunction",""), ("velocityz",""), ("temperature",""), ("vorticityz", "")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row == ("streamfunction","") or field_row == ("velocityz","") or field_row == ("temperature","") or field_row == ("vorticityz",""):
            fields = [("streamfunction",""), ("velocityz",""), ("temperature",""), ("vorticityz","")]
        else:
            fields = []

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("streamfunction",""), ("velocityz",""), ("temperature",""), ("vorticityz","")]:
                fields = [field_row]

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]*res[2]
        if self.use_galerkin:
            if field_row == ("temperature","") or field_row == ("velocityz",""):
                shift_x = 2
                shift_z = 0
            elif field_row == ("streamfunction",""):
                shift_x = 4
                shift_z = 0
            else:
                shift_x = 0
                shift_z = 0

            gal_n = (res[0] - shift_x)*(res[2] - shift_z)

        else:
            gal_n = tau_n
            shift_x = 0
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_x,0,shift_z), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex
        is_complex = True

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_SINGLE_RHS

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            tanchi = np.tan(eq_params['chi']*np.pi/180)
            G = eq_params['gamma']
            k = eigs[0]

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            if bcId == 0:
                if self.use_galerkin:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-40, 'rt':0}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-40, 'rt':0}, 'z':{0:10, 'c':-1j*k*tanchi/G}}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-20}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-20, 'rt':0}, 'z':{0:10}}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-40, 'rt':0}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-40, 'rt':0}, 'z':{0:11, 'c':1j*k*tanchi/G}}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-20}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-20, 'rt':0}, 'z':{0:11}}
                    elif field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}

                else:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:40}, 'z':{0:10, 'c':-1j*k*tanchi/G}}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:10}}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'z':{0:11}}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:0}, 'z':{0:11, 'c':1j*k*tanchi/G}}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'z':{0:0}}

            elif bcId == 1:
                if self.use_galerkin:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-41, 'rt':0}, 'z':{0:10, 'c':-1j*k*tanchi/G}}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-21}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-21}, 'z':{0:10}}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-41, 'rt':0}, 'z':{0:11, 'c':1j*k*tanchi/G}}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-21}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-21}, 'z':{0:11}}
                    elif field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:0}}
                    elif field_row == ("temperature",""):
                        bc = {'x':{0:-21}, 'z':{0:0}}
                else:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:41}, 'z':{0:10, 'c':-1j*k*tanchi/G}}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:10}}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:21}, 'z':{0:11}}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:0}, 'z':{0:11, 'c':1j*k*tanchi/G}}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'z':{0:0}}

            elif bcId == 2:
                if self.use_galerkin:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        bc = {'x':{0:-20}, 'z':{0:0}}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-41, 'rt':0}, 'z':{0:11, 'c':1j*k*tanchi/G}}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {'x':{0:-20}, 'z':{0:0}}
                        else:
                            bc = {'x':{0:-20}, 'z':{0:11}}
                    elif field_row == ("vorticityz","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:-41}, 'z':{0:10, 'c':-1j*k*tanchi/G}}
                    elif field_row == ("vorticityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:-20}, 'z':{0:10}}
                    elif field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("vorticityz",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}
                else:
                    if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:20}, 'z':{0:0}}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:00}}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'z':{0:11}}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:0}, 'z':{0:11, 'c':1j*k*tanchi/G}}
                    elif field_row == ("vorticityz","") and field_col == ("vorticityz",""):
                        bc = {'x':{0:20}, 'z':{0:0}}
                    elif field_row == ("vorticityz","") and field_col == ("streamfunction",""):
                        bc = {'x':{0:0}, 'z':{0:10, 'c':-1j*k*tanchi/G}}
                    elif field_row == ("vorticityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:10}}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['x']['rt'] = 4
                    bc['z']['rt'] = 0
                elif field_row == ("velocityz",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 0
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 0
                elif field_row == ("vorticity",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 0

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-40, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}

                elif bcId == 1:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:0}}

                elif bcId == 2:
                    if field_col == ("streamfunction",""):
                        bc = {'x':{0:-41, 'rt':0}, 'z':{0:0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:0}}
        
        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['x']['rt'] = 4
                elif field_row == ("velocityz",""):
                    bc['x']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                elif field_row == ("vorticityz",""):
                    bc['x']['rt'] = 2

        else:
            bc = no_bc()

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the quasi-inverse operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction","") and field_col == field_row:
            mat = geo.i2(res[0],res[2], bc)

        elif field_row == ("velocityz","") and field_col == field_row:
            mat = geo.i2j1(res[0],res[2], bc)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2(res[0],res[2], bc)

        elif field_row == ("vorticityz","") and field_col == field_row:
            mat = geo.i2j1(res[0],res[2], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block of linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        G = eq_params['gamma']
        tanchi = np.tan(eq_params['chi']*np.pi/180)

        xscale = eq_params['scale1d']
        zscale = eq_params['scale3d']

        k = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.i2laplh(res[0],res[2], k, bc, -1.0, xscale = xscale)

            elif field_col == ("velocityz",""):
                mat = geo.zblk(res[0],res[2], 2, 0, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0],res[2], 2, 0, bc)

            elif field_col == ("vorticityz",""):
                mat = geo.i2(res[0],res[2], bc)

        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.i2j1e1(res[0],res[2], bc, (-1.0/G**2), zscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.i2j1laplh(res[0],res[2], k, bc, xscale = xscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0],res[2], 2, 1, bc)

            elif field_col == ("vorticityz",""):
                mat = geo.zblk(res[0],res[2], 2, 1, bc)

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                if self.linearize:
                    mat = geo.i2(res[0],res[2], bc, 1j*k)
                else:
                    mat = geo.zblk(res[0],res[2], 2, 0, bc)

            elif field_col == ("velocityz",""):
                mat = geo.zblk(res[0],res[2], 2, 0, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2laplh(res[0],res[2],k, bc, (1/Pr), xscale = xscale)

            elif field_col == ("vorticityz",""):
                mat = geo.zblk(res[0],res[2], 2, 0, bc)

        elif field_row == ("vorticityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.zblk(res[0],res[2], 2, 1, bc)

            elif field_col == ("velocityz",""):
                mat = geo.i2j1e1(res[0],res[2], bc, zscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.i2j1(res[0],res[2], bc, 1j*k*(Ra/(16.0*Pr)))

            elif field_col == ("vorticityz",""):
                mat = geo.i2j1laplh(res[0],res[2],k, bc, xscale = xscale)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.zblk(res[0],res[2], 2, 0, bc)

        elif field_row == ("velocityz",""):
            mat = geo.i2j1(res[0],res[2], bc)

        elif field_row == ("temperature",""):
            mat = geo.i2(res[0],res[2], bc)

        elif field_row == ("vorticityz",""):
            mat = geo.i2j1(res[0],res[2], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
