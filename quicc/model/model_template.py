"""Module provides the functions to generate the Template"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.something.something as geo
import quicc.base.base_model as base_model
from quicc.geometry.something.something_boundary import no_bc


class BoussinesqDynamoShellStd(base_model.BaseModel):
    """Class to setup the TEMPLATE"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return []

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return []

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields = []

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""
    
        # fields are only coupled to themselves
        fields = [field_row]

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

        # Tau matrix dimension
        tau_n = res[0]

        # Setup Galerkin sizes
        if self.use_galerkin:
            if field_row == ("field","component"):
                shift_z = 2
            
            # Galerkin matrix size
            gal_n = (res[0] - shift_z)

        # Setup Tau sizes
        else:
            gal_n = tau_n
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_z,0,0), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = False

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_MULTI_RHS

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
            if bcId == 42:
                if self.use_galerkin:
                    if field_col == ("field","component"):
                        bc = {0:-42, 'rt':0}

                else:
                    if field_row == ("field","component") and field_col == field_row:
                        bc = {0:42}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("field","component") and field_col == ("field2","component2"):
                    bc['rt'] = 4

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 42:
                    if field_col == ("field","component"):
                        bc = {0:-42, 'rt':}
        
        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("field","component"):
                    bc['rt'] = 4

        else:
            bc = no_bc()

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("field","component") and field_col == field_row:
            mat = geo.something()

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit nonlinear term"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("field","component") and field_col == field_row:
            mat = geo.something()

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for implicit operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("field","component") and field_col == ("field2","component2"):
            mat = geo.something()

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""
    
        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("field","component") and field_col == field_row:
            mat = geo.something()

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
