

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius as geo
import quicc.base.base_model as base_model
from quicc.geometry.spherical.shell_radius_boundary import no_bc
from quicc.model.boussinesq_couetteshell_base import BoussinesqCouetteShellBase, BoussinesqCouetteShellBaseConfig, BoussinesqCouetteShellBaseVisu
from quicc.model.boussinesq_couetteshell_std import BoussinesqCouetteShellExplicitBase

class BoussinesqFreeShellExplicitBase(BoussinesqCouetteShellExplicitBase):

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        sgn = np.sign(eq_params['rossby'])
        sgn = 1 if sgn == 0 else sgn
        ro = self.automatic_parameters(eq_params)['ro']
        ri = ro * eq_params['rratio']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])
        assert (eigs[0].is_integer())
        m = int(eigs[0])

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)

            # inner boundary
            if bcId == 0:
                if self.use_galerkin:
                    raise RuntimeError("Inhomogeneous boundary conditions cannot use Galerkin scheme!")

                else:
                    if field_row == ("velocity", "tor") and field_col == field_row:
                        bc = {0: 26, 'c': {'a': a, 'b': b}}
                    elif field_row == ("velocity", "pol") and field_col == field_row:
                        bc = {0: 43, 'c': {'a': a, 'b': b}}
            # outer boundary
            elif bcId == 1:
                if self.use_galerkin:
                    raise RuntimeError("Inhomogeneous boundary conditions cannot use Galerkin scheme!")

                else:
                    if field_row == ("velocity", "tor") and field_col == field_row:
                        bc = {0: 25, 'c': {'a': a, 'b': b}}
                    elif field_row == ("velocity", "pol") and field_col == field_row:
                        bc = {0: 42, 'c': {'a': a, 'b': b}}


            # Set LHS galerkin restriction
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot use Galerkin scheme!")

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot use Galerkin scheme!")

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot use Galerkin scheme!")

        else:
            bc = no_bc()

        return bc

class BoussinesqFreeShellStdConfig(BoussinesqCouetteShellBaseConfig):
    pass

class BoussinesqFreeShellStd(BoussinesqFreeShellExplicitBase, BoussinesqCouetteShellBaseConfig):
    pass


class BoussinesqCouetteShellStdVisu(BoussinesqCouetteShellBaseVisu, BoussinesqFreeShellStd):
    pass
