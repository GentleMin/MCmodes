"""Module provides the functions to generate the Boussinesq rotating Rayleigh-Benard convection in a 3D box (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_3d as c3d
import quicc.geometry.cartesian.cartesian_1d as c1d
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_3d import no_bc


class BoussinesqRRB3DBoxVC(base_model.BaseModel):
    """Class to setup the Boussinesq rotating Rayleigh-Benard convection in a 3D box (velocity-continuity formulation)"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "taylor", "zxratio", "yxratio"]

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

        tau_n = res[0]*res[1]*res[2]
        if self.use_galerkin:
            if field_row == ("velocityx","") or field_row == ("velocityy","")  or field_row == ("velocityz","") or field_row == ("pressure","") or field_row == ("temperature",""):
                shift_x = 2
                shift_y = 2
                shift_z = 2
            else:
                shift_x = 0
                shift_y = 0
                shift_z = 0

            gal_n = (res[0] - shift_x)*(res[1] - shift_y)*(res[2] - shift_z)

        else:
            gal_n = tau_n
            shift_x = 0
            shift_y = 0
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_x,shift_y,shift_z), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = False

        # Implicit field coupling
        im_fields = self.implicit_fields(field_row)
        # Additional explicit linear fields
        ex_fields = self.explicit_fields(field_row)

        # Index mode: 
        index_mode = self.SINGLE

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

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            # No-slip/No-slip/No-slip, Fixed temperature/Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'xz'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'xz'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'xz'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'xz'}

            # Stress-free/Stress-free/Stress-free, Fixed flux/Fixed flux/Fixed flux
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'y':{0:21}, 'z':{0:21}, 'priority':'xz'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:21}, 'priority':'yx'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:21}, 'y':{0:21}, 'z':{0:20}, 'priority':'zx'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'y':{0:21}, 'z':{0:21}, 'priority':'sx'}

            # Stress-free/No-slip/No-slip, Fixed flux/Fixed temperature/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'xz'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:20}, 'priority':'yz'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}

            # No-slip/Stress-free/Stress-free, Fixed temperature/Fixed flux/Fixed flux
            elif bcId == 3:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {'x':{0:20}, 'y':{0:21}, 'z':{0:21}, 'priority':'xz'}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:21}, 'priority':'xy'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'y':{0:21}, 'z':{0:20}, 'priority':'zx'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'y':{0:21}, 'z':{0:21}, 'priority':'xz'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityz",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}

                elif bcId == 1:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityy",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}

                elif bcId == 2:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-20, 'r':0}}

                elif bcId == 3:
                    if field_col == ("velocityx",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-20, 'r':0}, 'z':{0:-21, 'r':0}}
                    elif field_col == ("velocityz",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-20, 'r':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'r':0}, 'y':{0:-21, 'r':0}, 'z':{0:-21, 'r':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("velocityz",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['r'] = 2
                    bc['y']['r'] = 2
                    bc['z']['r'] = 2

        return bc

    def stencil(self, res, eq_params, eigs, bcs, field_row):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return c3d.stencil(res[0], res[2], bc)

    def qi(self, res, eq_params, eigs, bcs, field_row):
        """Create the quasi-inverse operator"""

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc)

        elif field_row == ("velocityy",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc)

        elif field_row == ("velocityz",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc)

        elif field_row == ("temperature",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc)

        return mat

    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        Ta = eq_params['taylor']

        zscale = eq_params['zxratio']
        yscale = eq_params['yxratio']

#        zero_u = c3d.zblk(res[0], res[1], res[2], 1,1,1, no_bc())
        zero_u = spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        zero_u = zero_u + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        zero_u = zero_u + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
#        zero_u = zero_u + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        idx_u = (np.ravel(zero_u.sum(axis=1)) > 0)
        zero_u = spsp.lil_matrix(zero_u.shape)
        zero_u[idx_u,idx_u] = 1

#        zero_v = c3d.zblk(res[0], res[1], res[2], 1,1,1, no_bc())
        zero_v = spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_v = zero_v + spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_v = zero_v + spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
#        zero_v = zero_v + spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_v = (np.ravel(zero_v.sum(axis=1)) > 0)
        zero_v = spsp.lil_matrix(zero_v.shape)
        zero_v[idx_v,idx_v] = 1

#        zero_w = c3d.zblk(res[0], res[1], res[2], 1,1,1, no_bc())
        zero_w = spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_w = zero_w + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_w = zero_w + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
#        zero_w = zero_w + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_w = (np.ravel(zero_w.sum(axis=1)) > 0)
        zero_w = spsp.lil_matrix(zero_w.shape)
        zero_w[idx_w,idx_w] = 1

        # Pressures looses all field entries
#        zero_p = c3d.zblk(res[0], res[1], res[2], 1,1,1, no_bc())
        zero_p = zero_u + zero_v + zero_w

        # Highest 3 2D faces
        zero_p = zero_p + spsp.kron(c1d.qid(res[1], res[1]-3, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-3, c1d.c1dbc.no_bc())))
        zero_p = zero_p + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-3, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-3, c1d.c1dbc.no_bc())))
        zero_p = zero_p + spsp.kron(c1d.qid(res[1], res[1]-3, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-3, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        
        # Constant pressure
        zero_p = zero_p + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))

        idx_p = (np.ravel(zero_p.sum(axis=1)) > 0)
        zero_p = spsp.lil_matrix(zero_p.shape)
        zero_p[idx_p,idx_p] = 1

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)
                mat[idx_u,:] = 0
                mat[:,idx_u] = 0
                mat = mat + zero_u

            elif field_col == ("velocityy",""):
                mat = c3d.i2j2k2(res[0], res[1], res[2], bc, Ta**0.5)

            elif field_col == ("velocityz",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = c3d.i2j2k2d1(res[0], res[1], res[2], bc, -1.0).tolil()
                mat[idx_p,:] = 0
                mat[:,idx_p] = 0

        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = c3d.i2j2k2(res[0], res[1], res[2], bc, -Ta**0.5)

            elif field_col == ("velocityy",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)
                mat[idx_v,:] = 0
                mat[:,idx_v] = 0
                mat = mat + zero_v

            elif field_col == ("velocityz",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = c3d.i2j2k2e1(res[0], res[1], res[2], bc, -1.0, yscale = yscale).tolil()
                mat[idx_p,:] = 0
                mat[:,idx_p] = 0

        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)
                mat[idx_w,:] = 0
                mat[:,idx_w] = 0
                mat = mat + zero_w

            elif field_col == ("temperature",""):
                mat = c3d.i2j2k2(res[0], res[1], res[2], bc, Ra/16.0).tolil()
                mat[idx_w,:] = 0

            elif field_col == ("pressure",""):
                mat = c3d.i2j2k2f1(res[0], res[1], res[2], bc, -1.0, zscale = zscale).tolil()
                mat[idx_p,:] = 0
                mat[:,idx_p] = 0

        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c3d.i2j2k2(res[0], res[1], res[2], bc).tolil()
                mat[:,idx_w] = 0

            elif field_col == ("temperature",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)

            elif field_col == ("pressure",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

        elif field_row == ("pressure",""):
            if field_col == ("velocityx",""):
                bc['x']['rt'] = 1
                bc['x']['cr'] = 1
                bc['y']['rt'] = 1
                bc['y']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                mat = c3d.i1j1k1d1(res[0]+1, res[1]+1, res[2]+1, bc).tolil()
                mat[:,idx_u] = 0
                mat[idx_p,:] = 0

            elif field_col == ("velocityy",""):
                bc['x']['rt'] = 1
                bc['x']['cr'] = 1
                bc['y']['rt'] = 1
                bc['y']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                mat = c3d.i1j1k1e1(res[0]+1, res[1]+1, res[2]+1, bc, yscale = yscale).tolil()
                mat[:,idx_v] = 0
                mat[idx_p,:] = 0

            elif field_col == ("velocityz",""):
                bc['x']['rt'] = 1
                bc['x']['cr'] = 1
                bc['y']['rt'] = 1
                bc['y']['cr'] = 1
                bc['z']['rt'] = 1
                bc['z']['cr'] = 1
                mat = c3d.i1j1k1f1(res[0]+1, res[1]+1, res[2]+1, bc, zscale = zscale).tolil()
                mat[:,idx_w] = 0
                mat[idx_p,:] = 0

            elif field_col == ("temperature",""):
                mat = c3d.zblk(res[0], res[1], res[2], 1, 1, 1, bc)

            elif field_col == ("pressure",""):
                mat = c3d.zblk(res[0], res[1], res[2], 1, 1, 1, bc).tolil()
                mat = mat + zero_p

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']

        zero_u = c3d.qid(res[0], res[1], res[2], 0, res[1]-1, res[2]-1, no_bc())
        idx_u = (np.ravel(zero_u.sum(axis=1)) == 1)
        zero_v = c3d.qid(res[0], res[1], res[2], res[0]-1, 0, res[2]-1, no_bc())
        idx_v = (np.ravel(zero_v.sum(axis=1)) == 1)
        zero_w = c3d.qid(res[0], res[1], res[2], res[0]-1, res[1]-1, 0, no_bc())
        idx_w = (np.ravel(zero_w.sum(axis=1)) == 1)
        idx_p = np.logical_or(np.logical_or(idx_u, idx_v),idx_w)

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc, 1.0/Pr).tolil()
            mat[idx_u,:] = 0
            mat[:,idx_u] = 0

        elif field_row == ("velocityy",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc, 1.0/Pr).tolil()
            mat[idx_v,:] = 0
            mat[:,idx_v] = 0

        elif field_row == ("velocityz",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc, 1.0/Pr).tolil()
            mat[idx_w,:] = 0
            mat[:,idx_w] = 0

        elif field_row == ("temperature",""):
            mat = c3d.i2j2k2(res[0], res[1], res[2], bc)

        elif field_row == ("pressure",""):
            mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

        return mat
