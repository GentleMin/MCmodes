"""Module provides the functions to generate the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius as couette_geo
import quicc.geometry.spherical.shell as geo
import quicc.base.base_model as base_model
from quicc.geometry.spherical.shell_radius_boundary import no_bc
from quicc.model.boussinesq_couetteshell_base import BoussinesqCouetteShellBase, BoussinesqCouetteShellBaseConfig, BoussinesqCouetteShellBaseVisu

class BoussinesqCouetteShellImplicitBase(BoussinesqCouetteShellBase):

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
        is_complex = True

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_SINGLE_RHS

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields = [("velocity", "tor"), ("velocity", "pol")]

        return fields

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        assert(eigs[0].is_integer())
        m = int(eigs[0])

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.rad.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocity","tor"):
            mat = geo.i2r2(res[0], res[1], m, a, b, bc, with_sh_coeff = 'laplh', l_zero_fix = 'set', restriction = restriction)

        elif field_row == ("velocity","pol"):
            mat = geo.i4r4lapl(res[0], res[1], m, a, b, bc, with_sh_coeff = 'laplh', l_zero_fix = 'set', restriction = restriction)


        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        E = eq_params['ekman']
        assert(eigs[0].is_integer())

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.rad.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        #TODO: determing if for this scheme, eigs[0] is m or l
        m = int(eigs[0])

        if field_row == ("velocity","tor"):
            if field_col == ("velocity","tor"):
                mat = geo.i2r2lapl(res[0], res[1], m, a, b, bc, E, with_sh_coeff = 'laplh', l_zero_fix = 'zero', restriction = restriction)
                bc[0] = min(bc[0], 0)
                mat = mat + 2*geo.i2r2(res[0], res[1], m, a, b, bc, 1j*m, l_zero_fix = 'zero', restriction = restriction)

            elif field_col == ("velocity","pol"):
                mat = 2*geo.i2r2coriolis(res[0], res[1], m, a, b, bc, -1., l_zero_fix = 'zero', restriction = restriction)

        elif field_row == ("velocity","pol"):
            if field_col == ("velocity","tor"):
                mat = 2*geo.i4r4coriolis(res[0], res[1], m, a, b, bc, 1, l_zero_fix = 'zero', restriction = restriction)

            elif field_col == ("velocity","pol"):
                mat = geo.i4r4lapl2(res[0], res[1], m, a, b, bc, E, with_sh_coeff = 'laplh', l_zero_fix = 'zero', restriction = restriction)
                bc[0] = min(bc[0], 0)
                mat = mat + 2*geo.i4r4lapl(res[0], res[1], m, a, b, bc, 1j*m, l_zero_fix = 'zero', restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        m = int(eigs[0])
        mat = geo.zblk(res[0], res[1], m, bc, l_zero_fix='zero', restriction=restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

class BoussinesqCouetteShellConfig(BoussinesqCouetteShellBaseConfig):
    pass

class BoussinesqCouetteShell(BoussinesqCouetteShellBaseConfig, BoussinesqCouetteShellImplicitBase):
    pass
"""
class BoussinesqCouetteShellVisu(BoussinesqCouetteShellBaseVisu, BoussinesqCouetteShell):
    pass
"""
