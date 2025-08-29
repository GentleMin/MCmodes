"""Module provides the functions to generate the Boussinesq convection in a rotating 2D box model (velocity-continuity-temperature)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_2d as c2d
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_2d import no_bc


class BoussinesqRotConv2DBoxVCT(base_model.BaseModel):
    """Class to setup the Boussinesq convection in a rotating 2D box model (velocity-continuity-temperature)"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["taylor", "prandtl", "rayleigh", "ro", "rratio"]

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocityx", "velocityz", "pressure", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocityx",""), ("velocityz",""), ("temperature",""), ("pressure","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocityx",""), ("velocityz",""), ("temperature",""), ("pressure","")]

        return fields

    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        fields = []

        return fields

    def block_size(self, res, field_row):
        """Create block size information"""

        tau_n = res[0]*res[2]
        if self.use_galerkin:
            if field_row == ("velocityx","") or ("velocityz","") or ("pressure","") or field_row == ("temperature",""):
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
        is_complex = False

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
                    if field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
#                    elif field_row == ("pressure","") and field_col == ("pressure",""):
#                        bc = {'x':{0:12, 'c':-1.0}, 'z':{0:12, 'c':-1.0}, 'priority':'sz'}
#                    elif field_row == ("pressure","") and field_col == ("velocityx",""):
#                        bc = {'x':{0:14}, 'z':{0:0}, 'priority':'sz'}
#                    elif field_row == ("pressure","") and field_col == ("velocityz",""):
#                        bc = {'x':{0:0}, 'z':{0:14}, 'priority':'sz'}

            if bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    if field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'z':{0:21}, 'priority':'x'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'z':{0:21}, 'priority':'sx'}
#                    elif field_row == ("pressure","") and field_col == ("pressure",""):
#                        bc = {'x':{0:21}, 'z':{0:21}, 'priority':'sx'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['x']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityz",""):
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

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = c2d.i2j2lapl(res[0], res[2]-1, 0, bc)

            elif field_col == ("velocityz",""):
                bc['z']['rb'] = 1
                bc['x']['cr'] = 1
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                bc['z']['rb'] = 1
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                bc['z']['rb'] = 1
                bc['x']['cr'] = 1
                bc['z']['cr'] = 1
                mat = c2d.i2j2d1d0(res[0], res[2], bc, -1.0)

        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                bc['z']['cr'] = 1
                bc['x']['rb'] = 1
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c2d.i2j2lapl(res[0]-1, res[2], 0, bc)

            elif field_col == ("temperature",""):
                bc['x']['rb'] = 1
                mat = c2d.i2j2(res[0], res[2], bc, Ra/16.0)

            elif field_col == ("pressure",""):
                bc['x']['rb'] = 1
                bc['x']['cr'] = 1
                bc['z']['cr'] = 1
                mat = c2d.i2j2d0d1(res[0], res[2], bc, -1.0)

        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                bc['z']['cr'] = 1
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocityz",""):
                bc['x']['cr'] = 1
                mat = c2d.i2j2(res[0], res[2], bc)

            elif field_col == ("temperature",""):
                mat = c2d.i2j2lapl(res[0], res[2], 0, bc)

            elif field_col == ("pressure",""):
                bc['x']['cr'] = 1
                bc['z']['cr'] = 1
                mat = c2d.zblk(res[0], res[2], 2, 2, bc)

        elif field_row == ("pressure",""):
            if field_col == ("velocityx",""):
                bc['x']['rt'] = 1
                bc['x']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 2
                bc['x']['rb'] = 1
                bc['z']['rb'] = 1
                mat = c2d.i1j1d1d0(res[0]+1, res[2]+1, bc)
                mat = mat.tolil()
                mat[0,:] = 0
                mat[-res[0]-2:-res[0],:] = 0
                mat[-2:,:] = 0
                mat = mat.tocoo()
#                mat = c2d.i1j1d1d0(res[0], res[2], bc)

            elif field_col == ("velocityz",""):
                bc['x']['rt'] = 1
                bc['x']['cr'] = 2
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                bc['x']['rb'] = 1
                bc['z']['rb'] = 1
                mat = c2d.i1j1d0d1(res[0]+1, res[2]+1, bc)
                mat = mat.tolil()
                mat[0,:] = 0
                mat[-res[0]-2:-res[0],:] = 0
                mat[-2:,:] = 0
                mat = mat.tocoo()
#                mat = c2d.i1j1d0d1(res[0], res[2], bc)

            elif field_col == ("temperature",""):
                bc['x']['rb'] = 1
                bc['z']['rb'] = 1
                mat = c2d.zblk(res[0], res[2], 0, 0, bc)
                mat = mat.tolil()
                mat[0,:] = 0
                mat[-res[0]-2:-res[0],:] = 0
                mat[-2:,:] = 0
                mat = mat.tocoo()

            elif field_col == ("pressure",""):
                mat = c2d.zblk(res[0]-1, res[2]-1, 0, 0, bc)
                mat = mat.tolil()
                mat[0,:] = 0
                mat[0,0] = 1
                mat[-res[0]-2:-res[0],:] = 0
                mat[-res[0]-2:-res[0],-res[0]-2:-res[0]] = spsp.eye(2)
                mat[-2:,:] = 0
                mat[-2:,-2:] = spsp.eye(2)
                mat = mat.tocoo()

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        Ta = eq_params['taylor']
        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = c2d.i2j2(res[0], res[2]-1, bc, 1.0/Pr)

        elif field_row == ("velocityz",""):
            mat = c2d.i2j2(res[0]-1, res[2], bc, 1.0/Pr)

        elif field_row == ("temperature",""):
            mat = c2d.i2j2(res[0], res[2], bc)

        elif field_row == ("pressure",""):
            mat = c2d.zblk(res[0]-1, res[2]-1, 1, 1, bc)

        return mat
