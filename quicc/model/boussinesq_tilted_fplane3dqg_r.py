"""Module provides the functions to generate the Boussinesq tilted F-Plane 3DQG model (restricted because of blowup)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class BoussinesqTiltedFPlane3DQG(base_model.BaseModel):
    """Class to setup the Boussinesq tilted F-Plane 3DQG model (restricted because of blowup)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "theta", "scale1d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("streamfunction",""), ("velocityz",""), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row in [("streamfunction",""), ("velocityz",""), ("temperature","")]:
            fields =  [("streamfunction",""), ("velocityz",""), ("temperature","")]

        else:
            fields = []

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit linear dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("temperature",""), ("dz_meantemperature","")]:
                fields = [field_row]
            elif field_row in [("streamfunction",""), ("velocityz","")]:
                fields = [("no_streamfunction",""), ("no_velocityz","")]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            if field_row in [("no_streamfunction",""), ("no_velocityz",""), ("no_vorticityz","")]:
                fields = [("streamfunction",""), ("velocityz","")]
            else:
                fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("velocityz",""):
                shift_x = 2
            elif field_row == ("temperature",""):
                shift_x = 2
            else:
                shift_x = 0

            gal_n = res[0] - shift_x 

        else:
            gal_n = tau_n
            shift_x = 0

        block_info = (tau_n, gal_n, (shift_x,0,0), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
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
            eta3 = np.cos(np.pi*eq_params['theta']/180)
            kx = eigs[0]
            ky = eigs[1]

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocityz",""):
                        bc = {0:-20, 'rt':0}

                else:
                    if bcs["bcType"] == self.SOLVER_HAS_BC:
                        if field_row == ("velocityz","") and field_col == field_row:
                            bc = {0:11}
                        elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                            bc = {0:10}
                        elif field_row == ("temperature","") and field_col == field_row:
                            bc = {0:991}
                    else:
                        bc = no_bc()
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityz","") or field_row == ("streamfunction",""):
                    bc['rt'] = 1
                elif field_row == ("temperature",""):
                    bc['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                if field_col == ("temperature",""):
                    bc = {0:-20, 'rt':0}
                elif field_col == ("velocityz",""):
                    bc = {0:-20, 'rt':0}
        
        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocityz",""):
                    bc['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['rt'] = 2

        else:
            bc = no_bc()

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("no_streamfunction",""):
                bc['zb'] = 1
                mat = geo.i1(res[0], bc, eta3)

            elif field_col == ("no_velocityz",""):
                bc['zb'] = 1
                mat = geo.i1(res[0], bc, 1j*eta2*kx)

        elif field_row == ("velocityz",""):
            if field_col == ("no_streamfunction",""):
                if kx == 0 and ky == 0:
                    mat = geo.zblk(res[0], bc)
                else:
                    mat = geo.i1(res[0], bc, -1j*eta2*kx/(kx**2 + (1.0/eta3**2)*ky**2))

            elif field_col == ("no_velocityz",""):
                mat = geo.i1(res[0], bc, eta3)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.sid(res[0], 1, bc)

        elif field_row == ("dz_meantemperature","") and field_col == field_row:
            if eigs[0] == 0 and eigs[1] == 0:
                mat = (geo.qid(res[0],0, bc) - spsp.eye(res[0], 1)*geo.avg(res[0]))
            else:
                mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nextstep_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit nextstep update"""

        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        zscale = eq_params['scale1d']
        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("no_streamfunction",""):
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0], bc)
            elif field_col == ("streamfunction",""):
                mat = geo.qid(res[0],0, bc, eta3*(kx**2 + (1.0/eta3**2)*ky**2)/(kx**2 + ky**2))

            elif field_col == ("velocityz",""):
                mat = geo.qid(res[0],0, bc, 1j*eta2*kx/(kx**2 + ky**2))

        elif field_row == ("no_velocityz",""):
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0], bc)
            elif field_col == ("streamfunction",""):
                mat = geo.qid(res[0],0, bc, 1j*eta2*kx*(kx**2 + (1.0/eta3**2)*ky**2)/(kx**2 + ky**2))

            elif field_col == ("velocityz",""):
                mat = geo.qid(res[0],0, bc, eta3*(kx**2 + (1.0/eta3**2)*ky**2)/(kx**2 + ky**2))

        elif field_row == ("no_vorticityz",""):
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0], bc)
            elif field_col == ("streamfunction",""):
                mat = geo.qid(res[0],0, bc, -eta3*(kx**2 + (1.0/eta3**2)*ky**2)**2/(kx**2 + ky**2))

            elif field_col == ("velocityz",""):
                mat = geo.qid(res[0],0, bc, -1j*eta2*kx*(kx**2 + (1.0/eta3**2)*ky**2)/(kx**2 + ky**2))

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        zscale = eq_params['scale1d']
        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.i1(res[0], bc, (kx**2 + (1.0/eta3**2)*ky**2)**2)

            elif field_col == ("velocityz",""):
                mat = geo.i1d1(res[0], bc, eta3, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.i1d1(res[0], bc, -eta3, cscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.i1(res[0], bc, -(kx**2 + (1.0/eta3**2)*ky**2))

            elif field_col == ("temperature",""):
                if kx == 0 and ky == 0:
                    mat = geo.zblk(res[0], bc)
                else:
                    mat = geo.i1(res[0], bc, (Ra/Pr)*(kx**2 + ky**2)/(kx**2 + (1.0/eta3**2)*ky**2))

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocityz",""):
                if self.linearize:
                    mat = geo.sid(res[0],1, bc)
                else:
                    mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.sid(res[0],1, bc, -(1.0/Pr)*(kx**2 + (1.0/eta3**2)*ky**2))

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        eta3 = np.cos(np.pi*eq_params['theta']/180)
        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.i1(res[0], bc, -(kx**2 + (1.0/eta3**2)*ky**2))

        elif field_row == ("velocityz",""):
            mat = geo.i1(res[0], bc)

        elif field_row == ("temperature",""):
            mat = geo.sid(res[0],1, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
