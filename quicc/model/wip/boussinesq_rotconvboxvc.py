"""Module provides the functions to generate the Boussinesq convection in a rotating box model (velocity-continuity)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_2d as c2d
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_2d import no_bc


class BoussinesqRotConvBoxVC(base_model.BaseModel):
    """Class to setup the Boussinesq convection in a rotating box model (velocity-continuity)"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["taylor", "prandtl", "rayleigh", "ro", "rratio"]

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocityx", "velocityy", "velocityz", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocityx",""), ("velocityy",""), ("velocityz",""), ("temperature",""), ("pressure","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocityx",""), ("velocityy",""), ("velocityz",""), ("temperature",""), ("pressure","")]

        return fields

    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        fields = []

        return fields

    def block_size(self, res, field_row):
        """Create block size information"""

        tau_n = res[0]*res[2]
        if self.use_galerkin:
            if field_row == ("velocityx","") or field_row == ("velocityy","") or field_row == ("velocityz","") or field_row == ("temperature",""):
                shift_x = 2
                shift_z = 2
            else:
                shift_x = 0
                shift_z = 0

            gal_n = (res[0] - shift_x)*(res[2] - shift_z)

        else:
            gal_n = tau_n
            shift_x = 0
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_x,0,shift_z), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = True

        # Implicit field coupling
        im_fields = self.implicit_fields(field_row)
        # Additional explicit linear fields
        ex_fields = self.explicit_fields(field_row)

        # Index mode: SLOWEST = 0, MODE = 1
        index_mode = self.SLOWEST

        # Compute block info
        block_info = self.block_size(res, field_row)

        # Compute system size
        sys_n = 0
        for f in im_fields:
            sys_n += self.block_size(res, f)[1]
        
        if sys_n == 0:
            sys_n = block_info[1]
        block_info = block_info + (sys_n,)

        return (is_complex, im_fields, ex_fields, index_mode, block_info)

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            m = eigs[0]

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            # No-slip + velocity conditions
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-40, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
#                    elif field_row == ("pressure","") and field_col == ("velocityx",""):
#                        bc = {'x':{0:14}, 'z':{0:0}, 'priority':'r'}
#                    elif field_row == ("pressure","") and field_col == ("velocityz",""):
#                        bc = {'x':{0:0}, 'z':{0:14}, 'priority':'r'}

            # Stress-free + velocity conditions
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-40, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'z':{0:21}, 'priority':'x'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:21}, 'z':{0:21}, 'priority':'z'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'z':{0:21}, 'priority':'x'}
                    elif field_row == ("pressure","") and field_col == ("velocityx",""):
                        bc = {'x':{0:14}, 'z':{0:0}, 'priority':'r'}
                    elif field_row == ("pressure","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:14}, 'priority':'r'}

            # No-slip + pressure conditions
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-40, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("pressure","") and field_col == ("velocityx",""):
                        bc = {'x':{0:14}, 'z':{0:0}, 'priority':'r'}
                    elif field_row == ("pressure","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:14}, 'priority':'r'}

            # Stress-free + pressure conditions
            elif bcId == 3:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-40, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'z':{0:21}, 'priority':'x'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:21}, 'z':{0:21}, 'priority':'z'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'z':{0:21}, 'priority':'x'}
                    elif field_row == ("pressure","") and field_col == ("velocityx",""):
                        bc = {'x':{0:14}, 'z':{0:0}, 'priority':'r'}
                    elif field_row == ("pressure","") and field_col == ("velocityz",""):
                        bc = {'x':{0:0}, 'z':{0:14}, 'priority':'r'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_ror == ("velocityz",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'x':0}, 'z':{0:-20, 'x':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'x':0}, 'z':{0:-20, 'x':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'x':0}, 'z':{0:-20, 'x':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'x':0}, 'z':{0:-20, 'x':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityz",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2

        return bc

    def stencil(self, res, eq_params, eigs, bcs, field_row):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return c2d.stencil(res[0], res[2], bc)

    def qi(self, res, eq_params, eigs, bcs, field_row):
        """Create the quasi-inverse operator"""

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = c2d.i2j2(res[0], res[2], bc)

        elif field_row == ("velocityy",""):
            mat = c2d.i2j2(res[0], res[2], bc)

        elif field_row == ("velocityz",""):
            mat = c2d.i2j2(res[0], res[2], bc)

        elif field_row == ("temperature",""):
            mat = c2d.i2j2(res[0], res[2], bc)

        return mat

    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block linear operator"""

        Ta = eq_params['taylor']
        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        m = eigs[0]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = c2d.i2j2lapl(res[0], res[2], m, bc, 1.0)

            elif field_col == ("velocityy",""):
                mat = c2d.i2j2(res[0], res[2], bc, Ta**0.5)

            elif field_col == ("velocityz",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = c2d.i2j2d1d0(res[0], res[2], bc, -1.0)

            elif field_col == ("temperature",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = c2d.i2j2(res[0], res[2], bc, -Ta**0.5)

            elif field_col == ("velocityy",""):
                mat = c2d.i2j2lapl(res[0], res[2], m, bc, 1.0)

            elif field_col == ("velocityz",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = c2d.i2j2(res[0], res[2], bc, -1j*m)

            elif field_col == ("temperature",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c2d.i2j2lapl(res[0], res[2], m, bc, 1.0)

            elif field_col == ("pressure",""):
                mat = c2d.i2j2d0d1(res[0], res[2], bc, -1.0)

            elif field_col == ("temperature",""):
                mat = c2d.i2j2(res[0], res[2], bc, Ra)

        elif field_row == ("pressure",""):
            if field_col == ("velocityx",""):
                mat = c2d.i1j1d1d0(res[0], res[2], bc)

            elif field_col == ("velocityy",""):
                mat = c2d.i1j1(res[0], res[2], bc, 1j*m)

            elif field_col == ("velocityz",""):
                mat = c2d.i1j1d0d1(res[0], res[2], bc)

            elif field_col == ("pressure",""):
                mat = c2d.zblk(res[0], res[2], 1, 1, bc)

            elif field_col == ("temperature",""):
                mat = c2d.zblk(res[0], res[2], 1, 1, bc)

        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c2d.i2j2(res[0], res[2], bc)

            elif field_col == ("pressure",""):
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = c2d.i2j2lapl(res[0], res[2], m, bc)

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        Ta = eq_params['taylor']
        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        m = eigs[0]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = c2d.i2j2(res[0], res[2], bc, 1.0/Pr)

        elif field_row == ("velocityy",""):
            mat = c2d.i2j2(res[0], res[2], bc, 1.0/Pr)

        elif field_row == ("velocityz",""):
            mat = c2d.i2j2(res[0], res[2], bc, 1.0/Pr)

        elif field_row == ("pressure",""):
            mat = c2d.zblk(res[0], res[2], 2, 2, bc)

        elif field_row == ("temperature",""):
            mat = c2d.i2j2(res[0], res[2], bc)

        return mat
