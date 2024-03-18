"""Module provides the functions to generate the cylindrical Laplacian eigenvalues"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import functools

import quicc.base.utils as utils
import quicc.geometry.cylindrical.cylinder_worland as geo
import quicc.base.base_model as base_model
from quicc.geometry.cylindrical.cylinder_boundary_worland import no_bc


class BoussinesqIWCylinder(base_model.BaseModel):
    """Class to setup the Boussinesq inertial waves in a cylinder (Toroidal/Poloidal formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["gamma"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("temperature","")]

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

        tau_n = res[0]*res[2]
        if self.use_galerkin:
            if field_row == ("temperature",""):
                shift_r = 1
                shift_z = 2
                #shift_r = 2
                #shift_z = 4
            else:
                shift_r = 0
                shift_z = 0

            gal_n = (res[0] - shift_r)*(res[2] - shift_z)

        else:
            gal_n = tau_n
            shift_r = 0
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_r,0,shift_z), 1)
        return block_info

    def stencil(self, res, eq_params, eigs, bcs, field_row, make_square):
        """Create the galerkin stencil"""
        
        assert(eigs[0].is_integer())

        m = int(eigs[0])

        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return geo.stencil(res[0], res[2], m, bc, make_square)

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = False

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
            m = eigs[0]
            G = eq_params['gamma']
            zscale = 2.0*G

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            # No-slip/No-slip, Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_row == ("temperature","") and field_col == field_row:
                        bc = {'r':{0:-10}, 'z':{0:-20}}
                        #bc = {'r':{0:-20}, 'z':{0:-40}}

                else:
                    if field_row == ("temperature","") and field_col == field_row:
                        bc = {'r':{0:10}, 'z':{0:20}, 'priority':'z'}
                        #bc = {'r':{0:20}, 'z':{0:40}, 'priority':'z'}

            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    raise RuntimeError("Galerkin is not possible")

                else:
                    raise RuntimeError("Stress-free not implemented")
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("temperature",""):
                    bc['r']['rt'] = 1
                    bc['z']['rt'] = 2
                    #bc['r']['rt'] = 2
                    #bc['z']['rt'] = 4

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("temperature","tor"):
                        bc = {'r':{0:-10, 'rt':0}, 'z':{0:-20, 'rt':0}}
                        #bc = {'r':{0:-20, 'rt':0}, 'z':{0:-40, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("temperature",""):
                    bc['r']['rt'] = 1
                    bc['z']['rt'] = 2
                    #bc['r']['rt'] = 2
                    #bc['z']['rt'] = 4

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        m = eigs[0]

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        G = eq_params['gamma']
        m = eigs[0]

        zscale = 2.0*G

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("temperature","") and field_col == field_row:
            #mat = geo.i2j2lapl(res[0], res[2], m, bc, zscale = zscale)
            mat = geo.i4j4lapl2(res[0], res[2], m, bc, zscale = zscale)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        G = eq_params['gamma']
        m = eigs[0]

        zscale = 2.0*G

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("temperature",""):
            mat = geo.i2j2(res[0], res[2], m, bc, 1.0)
            #mat = geo.i4j4(res[0], res[2], m, bc, 1.0)
            #mat = geo.i4j4lapl(res[0], res[2], m, bc, 1.0)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
