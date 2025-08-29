"""Module provides the functions to generate the test model for the FFF scheme"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_0d as c0d
import quicc.base.base_model as base_model


class TestFFF(base_model.BaseModel):
    """Class to setup the test model for the FFF scheme"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "gamma", "chi"]


    def periodicity(self):
        """Get the domain periodicity"""

        return [True, True, True]


    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature"]


    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields = [("streamfunction",""), ("velocityz",""), ("temperature","")]

        return fields


    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        # Solve as coupled equations
        if True:
            fields = [("streamfunction",""), ("velocityz",""), ("temperature","")]

        # Solve as splitted equations
        else:
            fields = [field_row]

        return fields


    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        return []


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
        block_info = (1, 1)

        return (is_complex,im_fields,ex_fields,has_geometric_coupling, index_mode, block_info)


    def qi(self, res, eq_params, eigs, bcs, field_row):
        """Create the quasi-inverse operator"""

        if field_row == ("streamfunction",""):
            mat = c0d.qid()

        elif field_row == ("velocityz",""):
            mat = c0d.qid()

        elif field_row == ("temperature",""):
            mat = c0d.qid()

        return mat


    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block of linear operator"""

        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = c0d.lapl(eigs[0],eigs[1],eigs[2])

            elif field_col == ("velocityz",""):
                mat = c0d.zblk()

            elif field_col == ("temperature",""):
                mat = c0d.zblk()

        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = c0d.zblk(0)

            elif field_col == ("velocityz",""):
                mat = c0d.lapl2(eigs[0],eigs[1],eigs[2])

            elif field_col == ("temperature",""):
                mat = c0d.zblk()

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                mat = c0d.zblk()

            elif field_col == ("velocityz",""):
                mat = c0d.zblk()

            elif field_col == ("temperature",""):
                mat = c0d.lapl(eigs[0],eigs[1],eigs[2])

        return mat


    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        if field_row == ("streamfunction",""):
            mat = c0d.qid()

        elif field_row == ("velocityz",""):
            mat = c0d.lapl(eigs[0],eigs[1],eigs[2])

        elif field_row == ("temperature",""):
            mat = c0d.qid()

        return mat
