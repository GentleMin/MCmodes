"""Module provides the functions to generate the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius as geo
import quicc.base.base_model as base_model
import quicc.projection.shell as proj
from quicc.geometry.spherical.shell_radius_boundary import no_bc
from quicc.model.boussinesq_couetteshell import BoussinesqCouetteShellConfig, BoussinesqCouetteShellImplicitBase, BoussinesqCouetteShellBaseVisu


class BoussinesqNuttatingCouetteShellConfig(BoussinesqCouetteShellConfig):
    """ define the base methods that depend on the problem and not on the parallelization settings"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["ekman", "rossby", "omega", "rratio"]


class BoussinesqNuttatingCouetteShell(BoussinesqNuttatingCouetteShellConfig, BoussinesqCouetteShellImplicitBase):
    pass
"""
class BoussinesqNuttatingCouetteShellVisu(BoussinesqCouetteShellBaseVisu, BoussinesqNuttatingCouetteShell):
    pass
"""