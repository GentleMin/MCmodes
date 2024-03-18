"""Module provides the functions to generate the periodic Boussinesq Beta 3DQG model"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class BoussinesqBeta3DQGPer(base_model.BaseModel):
    """Class to setup the periodic Boussinesq Beta 3DQG model"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "gamma", "chi", "scale1d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature", "vorticityz"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields = [("streamfunction",""), ("velocityz",""), ("temperature",""), ("vorticityz", "")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row in [("streamfunction",""), ("velocityz",""), ("temperature",""), ("vorticityz","")]:
            fields = [("streamfunction",""), ("velocityz",""), ("temperature",""), ("vorticityz","")]
        else:
            fields = [field_row]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("velocityz",""), ("vorticityz",""), ("dx_meantemperature","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("temperature","") or field_row == ("velocityz",""):
                shift_x = 0
            elif field_row == ("streamfunction",""):
                shift_x = 0
            else:
                shift_x = 0

            gal_n = (res[0] - shift_x)

        else:
            gal_n = tau_n
            shift_x = 0

        block_info = (tau_n, gal_n, (shift_x,0,0), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex
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
            tanchi = np.tan(eq_params['chi']*np.pi/180)
            G = eq_params['gamma']
            kx = eigs[0]
            ky = eigs[1]

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            if bcId == 0:
                if self.use_galerkin:
                    if field_row == ("streamfunction","") and field_col == field_row:
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {0:0}
                        else:
                            bc = {0:10, 'c':-1j*ky*tanchi/G}
                    elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {0:0}
                        else:
                            bc = {0:10}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {0:0}
                        else:
                            bc = {0:11, 'c':1j*ky*tanchi/G}
                    elif field_row == ("velocityz","") and field_col == field_row:
                        if bcs["bcType"] == self.SOLVER_NO_TAU:
                            bc = {0:0}
                        else:
                            bc = {0:11}

                else:
                    if field_row == ("vorticityz","") and field_col == ("streamfunction",""):
                        bc = {0:10, 'c':-1j*ky*tanchi/G}
                    elif field_row == ("vorticityz","") and field_col == ("velocityz",""):
                        bc = {0:10}
                    elif field_row == ("velocityz","") and field_col == field_row:
                        bc = {0:11}
                    elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                        bc = {0:11, 'c':1j*ky*tanchi/G}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("streamfunction",""):
                    bc['rt'] = 0
                elif field_row == ("velocityz",""):
                    bc['rt'] = 0
                elif field_row == ("temperature",""):
                    bc['rt'] = 0
                elif field_row == ("vorticity",""):
                    bc['rt'] = 0

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("streamfunction",""):
                        bc = {0:0}
                    elif field_col == ("velocityz",""):
                        bc = {0:0}
                    elif field_col == ("temperature",""):
                        bc = {0:0}
        
        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()

        else:
            bc = no_bc()

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the quasi-inverse operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityz","") and field_col == field_row:
            mat = geo.i1(res[0], bc)

        elif field_row == ("vorticityz","") and field_col == field_row:
            mat = geo.i1(res[0], bc)

        elif field_row == ("dx_meantemperature","") and field_col == field_row:
            if eigs[1] == 0:
                mat = spsp.eye(res[0], 1)*geo.avg(res[0])
            else:
                mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block of linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        G = eq_params['gamma']
        tanchi = np.tan(eq_params['chi']*np.pi/180)
        zscale = eq_params['scale1d']

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.qid(res[0], 0, bc, (kx**2 + ky**2))

            elif field_col == ("velocityz",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("vorticityz",""):
                mat = geo.qid(res[0], 0, bc)

        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.i1d1(res[0], bc, (-1.0/G**2), cscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.i1(res[0], bc, -(kx**2 + ky**2))

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("vorticityz",""):
                mat = geo.zblk(res[0], bc)

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                if self.linearize:
                    mat = geo.qid(res[0], 0, bc, 1j*ky)
                else:
                    mat = geo.zblk(res[0], bc)

            elif field_col == ("velocityz",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.qid(res[0], 0, bc, -(1/Pr)*(kx**2 + ky**2))

            elif field_col == ("vorticityz",""):
                mat = geo.zblk(res[0], bc)

        elif field_row == ("vorticityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocityz",""):
                mat = geo.i1d1(res[0], bc, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.i1(res[0], bc, 1j*ky*(Ra/(16.0*Pr)))

            elif field_col == ("vorticityz",""):
                mat = geo.i1(res[0], bc, -(kx**2 + ky**2))

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.zblk(res[0], bc)

        elif field_row == ("velocityz",""):
            mat = geo.i1(res[0], bc)

        elif field_row == ("temperature",""):
            mat = geo.qid(res[0], 0, bc)

        elif field_row == ("vorticityz",""):
            mat = geo.i1(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
