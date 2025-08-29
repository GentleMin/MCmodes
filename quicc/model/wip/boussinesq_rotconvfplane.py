"""Module provides the functions to generate the Boussinesq convection in a rotating F-Plane model"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as c1d
import quicc.base.base_model as base_model


class BoussinesqRotConvFPlane(base_model.BaseModel):
    """Class to setup the Boussinesq convection in a rotating F-Plane model"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "taylor", "theta"]


    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]


    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocityx", "velocityy" ,"velocityz", "pressure", "temperature"]


    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields = [("velocityx",""), ("velocityy",""), ("velocityz",""), ("pressure",""), ("temperature","")]

        return fields


    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        return [("velocityx",""), ("velocityy",""), ("velocityz",""), ("pressure",""), ("temperature","")]


    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        fields = []

        return fields


    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
        is_complex = False

        # Implicit field coupling
        im_fields = self.implicit_fields(field_row)
        # Additional explicit linear fields
        ex_fields = self.explicit_fields(field_row)

        # Equation doesn't have geometric coupling
        has_geometric_coupling = False
        # Index mode: SLOWEST = 0, MODE = 1
        index_mode = 1

        # Rows per equation block and number of rhs
        block_info = (res[0], 1)

        return (is_complex, im_fields, ex_fields, has_geometric_coupling, index_mode, block_info)


    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        use_tau_boundary = True

        # Impose no boundary conditions
        no_bc = [0]
        if bcs["bcType"] == 2:
            bc = no_bc
        else:
            # Impose no boundary conditions
            if bcs["bcType"] == 1 and use_tau_boundary:
                bc = no_bc
            else: #bcType == 0 or Galerkin boundary
                eta2 = np.sin(np.pi*eq_params['theta']/180)
                eta3 = np.cos(np.pi*eq_params['theta']/180)
                kx = eigs[0]
                ky = eigs[1]

                bc = None
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    bc_field = {}
                    bc_field[("velocityx","")] = [20]
                    bc_field[("velocityy","")] = [20]
                    bc_field[("velocityz","")] = [20]
                    bc_field[("pressure","")] = [20]
                    bc_field[("temperature","")] = [20]
                    if field_col == field_row:
                        bc = bc_field[field_col]

                if bc is None:
                    if use_tau_boundary:
                        bc = no_bc
                    else:
                        bc = bc_field[field_col]
                        bc[0] = -bc[0]

        return bc


    def qi(self, res, eq_params, eigs, bcs, field_row):
        """Create the quasi-inverse operator"""

        if field_row == ("velocityx",""):
            mat = c1d.i2(res[0], [0])

        elif field_row == ("velocityy",""):
            mat = c1d.i2(res[0], [0])

        elif field_row == ("velocityz",""):
            mat = c1d.i2(res[0], [0])

        elif field_row == ("pressure",""):
            mat = c1d.i2(res[0], [0])

        elif field_row == ("temperature",""):
            mat = c1d.i2(res[0], [0])

        return mat


    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        Ta = eq_params['taylor']
        T = Ta**0.5
        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        kx = eigs[0]
        ky = eigs[1]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        # X-momentum
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = c1d.i2(res[0], bc, -(kx**2 + ky**2)) + c1d.i2d2(res[0], bc, 4.0)

            elif field_col == ("velocityy",""):
                mat = c1d.i2(res[0], bc, T*eta3)

            elif field_col == ("velocityz",""):
                mat = c1d.i2(res[0], bc, -T*eta2)

            elif field_col == ("pressure",""):
                mat = c1d.i2(res[0], bc, -1j*kx)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 2, bc)

        # Y-momentum
        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = c1d.i2(res[0], bc, -T*eta3)

            elif field_col == ("velocityy",""):
                mat = c1d.i2(res[0], bc, -(kx**2 + ky**2)) + c1d.i2d2(res[0], bc, 4.0)

            elif field_col == ("velocityz",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("pressure",""):
                mat = c1d.i2(res[0], bc, -1j*ky)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 2, bc)

        # Z-momentum
        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                mat = c1d.i2(res[0], bc, T*eta2)

            elif field_col == ("velocityy",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("velocityz",""):
                mat = c1d.i2(res[0], bc, -(kx**2 + ky**2)) + c1d.i2d2(res[0], bc, 4.0)

            elif field_col == ("pressure",""):
                mat = c1d.i2d1(res[0], bc, 2.0)

            elif field_col == ("temperature",""):
                mat = c1d.i2(res[0], bc, (Ra/Pr))

        # Mass conservation
        elif field_row == ("pressure",""):
            if field_col == ("velocityx",""):
                mat = c1d.qid(res[0], 0, bc, 1j*kx)

            elif field_col == ("velocityy",""):
                mat = c1d.qid(res[0], 0, bc, 1j*ky)

            elif field_col == ("velocityz",""):
                mat = c1d.qid(res[0], 0, bc, -2.0)

            elif field_col == ("pressure",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 0, bc)

        # Energy
        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityy",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityz",""):
                mat = c1d.i2(res[0], bc)

            elif field_col == ("pressure",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("temperature",""):
                mat = c1d.i2(res[0], bc, -(1.0/Pr)*(kx**2 + ky**2)) + c1d.i2d2(res[0], bc, 4.0)

        return mat


    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        # X-momentum
        if field_row == ("velocityx",""):
            mat = c1d.i2(res[0], bc)

        # Y-momentum
        elif field_row == ("velocityy",""):
            mat = c1d.i2(res[0], bc)

        # Z-momentum
        elif field_row == ("velocityz",""):
            mat = c1d.i2(res[0], bc)

        # Thermodynamic relation
        elif field_row == ("pressure",""):
            mat = c1d.zblk(res[0], 0, bc)
        
        # Energy
        elif field_row == ("temperature",""):
            mat = c1d.i2(res[0], bc)

        return mat
