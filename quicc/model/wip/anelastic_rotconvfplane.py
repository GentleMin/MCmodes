"""Module provides the functions to generate the anelastic convection in a rotating F-Plane model"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as c1d
import quicc.base.base_model as base_model


class AnelasticRotConvFPlane(base_model.BaseModel):
    """Class to setup the anelastic convection in a rotating F-Plane model"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "taylor", "density_scales", "polytropic_index", "theta"]


    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]


    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocityx", "velocityy", "velocityz", "pressure", "density", "temperature", "entropy"]


    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocityx",""), ("velocityy",""), ("velocityz",""), ("pressure",""), ("density",""), ("temperature",""), ("entropy","")]

        return fields


    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocityx",""), ("velocityy",""), ("velocityz",""), ("pressure",""), ("density",""), ("temperature",""), ("entropy","")]

        return fields


    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        fields = []

        return fields


    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex
        is_complex = True

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
                kx = eigs[0]
                ky = eigs[1]

                bc = None
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    bc_field = {}
                    bc_field[("velocityx","")] = [21]
                    bc_field[("velocityy","")] = [21]
                    bc_field[("velocityz","")] = [20]
                    bc_field[("pressure","")] = [0]
                    bc_field[("density","")] = [0]
                    bc_field[("temperature","")] = [0]
                    bc_field[("entropy","")] = [0]
                    if field_col == field_row:
                        bc = bc_field[field_col]

                    if field_row == ("entropy","") and field_col == ("temperature",""):
                        bc = [20]

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

        elif field_row == ("temperature",""):
            mat = c1d.i2(res[0], [0])

        return mat


    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        Ta = eq_params['taylor']
        n_rho = eq_params['density_scales']
        n = eq_params['polytropic_index']
        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        gamma = 1/n + 1
        kx = eigs[0]
        ky = eigs[1]

        h_t = np.exp(n_rho/n)-1.0;

        if Pr >= 1:
            # Free-fall scaling
            c1 = (Pr*Ta/Ra)**0.5
            c2 = (Pr/Ra)**0.5
            c4 = (Pr*Ra)**(-0.5)
            c5 = 1
        else:
            # Viscou scaling
            c1 = Ta**0.5
            c2 = 1
            c4 = 1/Pr
            c5 = Ra/Pr

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        # X-momentum
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = c1d.i2(res[0], bc, -c2*(kx**2 + ky**2 + (1.0/3.0)*kx**2)) + c1d.i2d2(res[0], bc, 4.0*c2)

            elif field_col == ("velocityy",""):
                mat = c1d.i2(res[0], bc, -c1*eta3) + c1d.i2(res[0], bc, -c2*(1.0/3.0)*kx*ky)

            elif field_col == ("velocityz",""):
                mat = c1d.i2d1(res[0], bc, 1j*c2*2.0*(1.0/3.0)*kx) + c1d.i2(res[0], bc, c1*eta2)

            elif field_col == ("pressure",""):
                mat = c1d.i2(res[0], bc, -1j*c5*kx)

            elif field_col == ("density",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("entropy",""):
                mat = c1d.zblk(res[0], 2, bc)

        # Y-momentum
        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = c1d.i2(res[0], bc, c1*eta3) + c1d.i2(res[0], bc, -c2*(1.0/3.0)*kx*ky)

            elif field_col == ("velocityy",""):
                mat = c1d.i2(res[0], bc, -c2*(kx**2 + ky**2 + (1.0/3.0)*ky**2)) + c1d.i2d2(res[0], bc, 4.0*c2)

            elif field_col == ("velocityz",""):
                mat = c1d.i2d1(res[0], bc, 1j*c2*2.0*(1.0/3.0)*ky)

            elif field_col == ("pressure",""):
                mat = c1d.i2(res[0], bc, -1j*c5*ky)

            elif field_col == ("density",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("entropy",""):
                mat = c1d.zblk(res[0], 2, bc)

        # Z-momentum
        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                mat = c1d.i2(res[0], bc, -c1*eta2) + c1d.i2d1(res[0], bc, 1j*c2*2*(1.0/3.0)*kx)

            elif field_col == ("velocityy",""):
                mat = c1d.i2d1(res[0], bc, 1j*c2*2.0*(1.0/3.0)*ky)

            elif field_col == ("velocityz",""):
                mat = c1d.i2(res[0], bc, -c2*(kx**2 + ky**2)) + c1d.i2d2(res[0], bc, c2*4.0*4.0/3.0)

            elif field_col == ("pressure",""):
                mat = c1d.i2d1(res[0], bc, -2.0*c5)

            elif field_col == ("density",""):
                mat = c1d.i2(res[0], bc, c5)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 2, bc)

            elif field_col == ("entropy",""):
                mat = c1d.zblk(res[0], 2, bc)

        # Thermodynamic relation
        elif field_row == ("pressure",""):
            if field_col == ("velocityx",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityy",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityz",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("pressure",""):
                mat = c1d.qid(res[0], 0, bc, 1.0/gamma)

            elif field_col == ("density",""):
                mat = c1d.qid(res[0], 0, bc, -1.0)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("entropy",""):
                mat = c1d.qid(res[0], 0, bc, -1.0)

        # Mass conservation
        elif field_row == ("density",""):
            if field_col == ("velocityx",""):
                mat = c1d.qid(res[0], 0, bc, 1j*kx)

            elif field_col == ("velocityy",""):
                mat = c1d.qid(res[0], 0, bc, 1j*ky)

            elif field_col == ("velocityz",""):
                mat = c1d.qid(res[0], 0, bc, 2.0)

            elif field_col == ("pressure",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("density",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("temperature",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("entropy",""):
                mat = c1d.zblk(res[0], 0, bc)

        # Equation of state
        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityy",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityz",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("pressure",""):
                mat = c1d.qid(res[0], 0, bc)

            elif field_col == ("density",""):
                mat = c1d.qid(res[0], 0, bc, -1.0)

            elif field_col == ("temperature",""):
                mat = c1d.qid(res[0], 0, bc, -1.0)

            elif field_col == ("entropy",""):
                mat = c1d.zblk(res[0], 0, bc)

        # Energy
        elif field_row == ("entropy",""):
            if field_col == ("velocityx",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityy",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("velocityz",""):
                mat = c1d.i2(res[0], bc, -1.0)

            elif field_col == ("pressure",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("density",""):
                mat = c1d.zblk(res[0], 0, bc)

            elif field_col == ("temperature",""):
                mat = c1d.i2(res[0], bc, -c4*(kx**2 + ky**2)) + c1d.i2d2(res[0], bc, c4*4.0)

            elif field_col == ("entropy",""):
                mat = c1d.zblk(res[0], 0, bc)

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
        
        # Mass conservation
        elif field_row == ("density",""):
            mat = c1d.zblk(res[0], 0, bc)
        
        # Equation of state
        elif field_row == ("temperature",""):
            mat = c1d.zblk(res[0], 0, bc)
    
        # Energy
        elif field_row == ("entropy",""):
            mat = c1d.i2(res[0], bc)

        return mat
