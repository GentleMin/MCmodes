"""Module provides the functions to generate the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius as geo
import quicc.base.base_model as base_model
from quicc.geometry.spherical.shell_radius_boundary import no_bc
from quicc.model.boussinesq_couetteshell_base import BoussinesqCouetteShellBase, BoussinesqCouetteShellBaseConfig, BoussinesqCouetteShellBaseVisu



class BoussinesqCouetteShellExplicitBase(BoussinesqCouetteShellBase):

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
        is_complex = False

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_MULTI_RHS

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields = [field_row]

        return fields

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction=None):
        """Create matrix block of time operator"""

        assert (eigs[0].is_integer())
        l = eigs[0]

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params, eigs, bcs, field_row, field_row)
        if field_row == ("velocity", "tor"):
            mat = geo.i2r2(res[0], a, b, bc, l * (l + 1.0))

        elif field_row == ("velocity", "pol"):
            mat = geo.i4r4lapl(res[0], l, a, b, bc, l * (l + 1.0))

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        E = eq_params['ekman']
        assert(eigs[0].is_integer())
        l = eigs[0]

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","tor") and field_col == field_row:
            mat = geo.i2r2lapl(res[0], l, a, b, bc, l*(l+1.0)*E)

        elif field_row == ("velocity","pol") and field_col == field_row:
            mat = geo.i4r4lapl2(res[0], l, a, b, bc, l*(l+1.0)*E)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction=None):
        """Create matrix block linear operator"""

        mat = None
        bc = self.convert_bc(eq_params, eigs, bcs, field_row, field_col)
        mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

class BoussinesqCouetteShellStdConfig(BoussinesqCouetteShellBaseConfig):
    pass

class BoussinesqCouetteShellStd(BoussinesqCouetteShellExplicitBase, BoussinesqCouetteShellBaseConfig):
    pass


class BoussinesqCouetteShellStdVisu(BoussinesqCouetteShellBaseVisu, BoussinesqCouetteShellStd):
    pass
