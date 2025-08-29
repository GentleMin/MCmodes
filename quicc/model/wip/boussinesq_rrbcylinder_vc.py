"""Module provides the functions to generate the Boussinesq rotating Rayleigh-Benard convection in a cylinder (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cylindrical.cylinder as cylinder
import quicc.base.base_model as base_model
from quicc.geometry.cylindrical.cylinder_boundary import no_bc


class BoussinesqRRBCylinderVC(base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard rotating convection in a cylinder (velocity-continuity formulation)"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "taylor", "zrratio"]

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
                shift_r = 2
                shift_z = 2
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

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = True

        # Implicit field coupling
        im_fields = self.implicit_fields(field_row)
        # Additional explicit linear fields
        ex_fields = self.explicit_fields(field_row)

        # Index mode: 
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
            # No-slip/No-slip, Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'r':{0:10}, 'z':{0:20}, 'priority':'r'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'r':{0:10}, 'z':{0:20}, 'priority':'r'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'r':{0:10}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'r':{0:10}, 'z':{0:20}, 'priority':'z'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2
                elif field_row == ("velocityz",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2
                elif field_row == ("temperature",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocityx",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityx",""):
                        bc = {'r':{0:-10, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-11, 'r':0}, 'z':{0:-20, 'r':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2
                elif field_row == ("velocityz",""):
                    bc['r']['r'] = 1
                    bc['z']['r'] = 2
                elif field_row == ("temperature",""):
                    bc['r']['r'] = 1
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
            mat = cylinder.i2j2x2(res[0], res[2], m, bc)

        elif field_row == ("velocityy",""):
            mat = cylinder.i2j2x2(res[0], res[2], m, bc)

        elif field_row == ("velocityy",""):
            mat = cylinder.i2j2x2(res[0], res[2], m, bc)

        elif field_row == ("temperature",""):
            mat = cylinder.i2j2x2(res[0], res[2], m, bc)

        return mat

    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        Ta = eq_params['taylor']

        zscale = 2.0*eq_params['zrratio']

        m = eigs[0]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = cylinder.i2j2x2lapl(res[0], res[2], m, (m+1)%2, bc, zscale = zscale)
                bc['r'][0] = min(bc['r'][0], 0)
                bc['z'][0] = min(bc['z'][0], 0)
                mat = mat + cylinder.i2j2(res[0], res[2], (m+1)%2, bc, -1.0)

            elif field_col == ("velocityy",""):
                mat = cylinder.i2j2(res[0], res[2], (m+1)%2, bc, -2.0*1j*m)
                bc['r'][0] = min(bc['r'][0], 0)
                bc['z'][0] = min(bc['z'][0], 0)
                mat = mat + cylinder.i2j2x2(res[0], res[2], (m+1)%2, bc, Ta**0.5)

            elif field_col == ("velocityz",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc)

            elif field_col == ("temperature",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc)

            elif field_col == ("pressure",""):
                mat = cylinder.i2j2x2d1(res[0], res[2], m%2, bc, -1.0)

        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = cylinder.i2j2(res[0], res[2], (m+1)%2, bc, 2.0*1j*m)
                bc['r'][0] = min(bc['r'][0], 0)
                bc['z'][0] = min(bc['z'][0], 0)
                mat = mat + cylinder.i2j2x2(res[0], res[2], (m+1)%2, bc, -Ta**0.5)

            elif field_col == ("velocityy",""):
                mat = cylinder.i2j2x2lapl(res[0], res[2], m, (m+1)%2, bc, zscale = zscale)
                bc['r'][0] = min(bc['r'][0], 0)
                bc['z'][0] = min(bc['z'][0], 0)
                mat = mat + cylinder.i2j2(res[0], res[2], (m+1)%2, bc, -1.0)

            elif field_col == ("velocityz",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc)

            elif field_col == ("temperature",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc)

            elif field_col == ("pressure",""):
                mat = cylinder.i2j2x1(res[0], res[2], m%2, bc, -1j*m)

        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                mat = cylinder.zblk(res[0], res[2], (m+1)%2, 1, 2, bc)

            elif field_col == ("velocityy",""):
                mat = cylinder.zblk(res[0], res[2], (m+1)%2, 1, 2, bc)

            elif field_col == ("velocityz",""):
                mat = cylinder.i2j2x2lapl(res[0], res[2], m, m%2, bc, zscale = zscale)

            elif field_col == ("temperature",""):
                mat = cylinder.i2j2x2(res[0], res[2], m%2, bc, Ra)

            elif field_col == ("pressure",""):
                mat = cylinder.i2j2x2e1(res[0], res[2], m%2, bc, -1.0, zscale = zscale)

        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                mat = cylinder.zblk(res[0], res[2], (m+1)%2, 1, 2, bc)

            elif field_col == ("velocityy",""):
                mat = cylinder.zblk(res[0], res[2], (m+1)%2, 1, 2, bc)

            elif field_col == ("velocityz",""):
                mat = cylinder.i2j2x2(res[0], res[2], m%2, bc)

            elif field_col == ("temperature",""):
                mat = cylinder.i2j2x2lapl(res[0], res[2], m, m%2, bc, zscale = zscale)

            elif field_col == ("pressure",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc)

        elif field_row == ("pressure",""):
            # Remove 2 highest pressure modes
            zero_p = cylinder.qid(res[0], res[2], 0, res[0]-2, res[2]-2, no_bc())
            idx_p = (np.ravel(zero_p.sum(axis=1)) > 0)
        
            if field_col == ("velocityx",""):
                bc['r']['rt'] = 1
                bc['r']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                mat = cylinder.i1j1x1div(res[0]+1, res[2]+1, (m+1)%2, bc).tolil()
                mat[idx_p,:] = 0

            elif field_col == ("velocityy",""):
                bc['r']['rt'] = 1
                bc['r']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                mat = cylinder.i1j1(res[0]+1, res[2]+1, (m%1)%2, bc, 1j*m).tolil()
                mat[idx_p,:] = 0

            elif field_col == ("velocityz",""):
                bc['r']['rt'] = 1
                bc['r']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                mat = cylinder.i1j1x1e1(res[0]+1, res[2]+1, m%2, bc, zscale = zscale).tolil()
                mat[idx_p,:] = 0

            elif field_col == ("temperature",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc)

            elif field_col == ("pressure",""):
                mat = cylinder.zblk(res[0], res[2], m%2, 1, 2, bc).tolil()
                mat[idx_p,:] = 0
                mat = mat + zero_p

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']

        m = eigs[0]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = cylinder.i2j2x2(res[0], res[2], (m+1)%2, bc, 1.0/Pr)

        elif field_row == ("velocityy",""):
            mat = cylinder.i2j2x2(res[0], res[2], (m+1)%2, bc, 1.0/Pr)

        elif field_row == ("velocityz",""):
            mat = cylinder.i2j2x2(res[0], res[2], m%2, bc, 1.0/Pr)

        elif field_row == ("temperature",""):
            mat = cylinder.i2j2x2(res[0], res[2], m%2, bc)

        elif field_row == ("pressure",""):
            mat = cylinder.zblk(res[0], res[2], m%2, 1, 1, bc)

        return mat
