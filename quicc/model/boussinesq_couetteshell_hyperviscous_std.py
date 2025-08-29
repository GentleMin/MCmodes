"""Module provides the functions to generate the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius as geo
import quicc.base.base_model as base_model
from quicc.geometry.spherical.shell_radius_boundary import no_bc
from quicc.model.boussinesq_couetteshell_std import BoussinesqCouetteShellExplicitBase, BoussinesqCouetteShellBaseVisu, BoussinesqCouetteShellBaseConfig


class BoussinesqCouetteShellHyperviscousStdConfig(BoussinesqCouetteShellBaseConfig):
    pass

class BoussinesqCouetteShellHyperviscousStd(BoussinesqCouetteShellHyperviscousStdConfig, BoussinesqCouetteShellExplicitBase):

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction=None):
        """Create matrix block linear operator"""

        E = eq_params['ekman']
        assert (eigs[0].is_integer())
        l = eigs[0]

        """
        # applies a constant Ekman up to 25 and then let scale from 50 onward
        if l>=25:
            E = E * (l/25.)**2
        """
        lmax = res[1]
        l0 = np.ceil(lmax * 0.8)
        if l > l0:
            E = E * (3.2/E) ** ((l - l0) / (lmax - l0))

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params, eigs, bcs, field_row, field_col)
        if field_row == ("velocity", "tor") and field_col == field_row:
            mat = geo.i2r2lapl(res[0], l, a, b, bc, l * (l + 1.0) * E)

        elif field_row == ("velocity", "pol") and field_col == field_row:
            mat = geo.i4r4lapl2(res[0], l, a, b, bc, l * (l + 1.0) * E)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    pass


class BoussinesqCouetteShellHyperviscousStdVisu(BoussinesqCouetteShellBaseVisu, BoussinesqCouetteShellHyperviscousStd):
    pass

