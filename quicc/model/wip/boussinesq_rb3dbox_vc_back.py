"""Module provides the functions to generate the Boussinesq Rayleigh-Benard convection in a 3D box (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_3d as c3d
import quicc.geometry.cartesian.cartesian_2d as c2d
import quicc.geometry.cartesian.cartesian_1d as c1d
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_3d import no_bc


class BoussinesqRB3DBoxVC(base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a 3D box (velocity-continuity formulation)"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "zxratio", "yxratio"]

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
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'yx'}
                    elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}

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
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:21}, 'priority':'yz'}
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
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:21}, 'priority':'yx'}
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

        zscale = eq_params['zxratio']
        yscale = eq_params['yxratio']
    
        # U: T_iNN
        zero_ru = spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        # U: T_i0N
        zero_cu = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        # U: T_iN0
        zero_cu = zero_cu + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        # U: T_i00
        zero_cu = zero_cu + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], 0, c1d.c1dbc.no_bc())))
        idx_ru = (np.ravel(zero_ru.sum(axis=1)) > 0)
        idx_cu = (np.ravel(zero_cu.sum(axis=1)) > 0)
        zero_ru = spsp.lil_matrix(zero_ru.shape)
        zero_cu = spsp.lil_matrix(zero_cu.shape)
        zero_ru[idx_ru,idx_ru] = 1
        zero_cu[idx_cu,idx_cu] = 1
        # U: Handle overlaps T_000, T_N00, T_0N0, T_NN0, T_00N, T_N0N
        zero_rut = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_cu = zero_cu - zero_rut
        idx_rut = (np.ravel(zero_rut.sum(axis=1)) > 0)
        zero_rut = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-2, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-2, c1d.c1dbc.no_bc())))
        zero_rut = zero_rut + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-2, c1d.c1dbc.no_bc())))
        idx_cut = (np.ravel(zero_rut.sum(axis=1)) > 0)
        zero_rut = spsp.lil_matrix(zero_rut.shape)
        #zero_rut[idx_cut,idx_rut] = 1
        zero_rut[idx_rut,idx_rut] = 1

        # V: T_NjN
        zero_rv = spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # V: T_0jN
        zero_cv = spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # V: T_Nj0
        zero_cv = zero_cv + spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # V: T_0j0
        zero_cv = zero_cv + spsp.kron(c1d.qid(res[1], 0, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_rv = (np.ravel(zero_rv.sum(axis=1)) > 0)
        idx_cv = (np.ravel(zero_cv.sum(axis=1)) > 0)
        zero_rv = spsp.lil_matrix(zero_rv.shape)
        zero_cv = spsp.lil_matrix(zero_cv.shape)
        zero_rv[idx_rv,idx_rv] = 1
        zero_cv[idx_cv,idx_cv] = 1
        # V: Handle overlaps T_000, T_0N0, T_N00, T_NN0, T_00N, T_0NN
        zero_rvt = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_cv = zero_cv - zero_rvt
        idx_rvt = (np.ravel(zero_rvt.sum(axis=1)) > 0)
        zero_rvt = spsp.kron(c1d.sid(res[1], res[1]-2, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.sid(res[1], res[1]-2, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rvt = zero_rvt + spsp.kron(c1d.sid(res[1], res[1]-2, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_cvt = (np.ravel(zero_rvt.sum(axis=1)) > 0)
        zero_rvt = spsp.lil_matrix(zero_rvt.shape)
        #zero_rvt[idx_cvt,idx_rvt] = 1
        zero_rvt[idx_rvt,idx_rvt] = 1

        # W: T_NNk
        zero_rw = spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # W: T_N0k
        zero_cw = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # W: T_0Nk
        zero_cw = zero_cw + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # W: T_00k
        zero_cw = zero_cw + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],0, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_rw = (np.ravel(zero_rw.sum(axis=1)) > 0)
        idx_cw = (np.ravel(zero_cw.sum(axis=1)) > 0)
        zero_rw = spsp.lil_matrix(zero_rw.shape)
        zero_cw = spsp.lil_matrix(zero_cw.shape)
        zero_rw[idx_rw,idx_rw] = 1
        zero_cw[idx_cw,idx_cw] = 1
        # W: Handle overlaps T_000, T_00N, T_N00, T_N0N, T_0N0, T_0NN
        zero_rwt = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_cw = zero_cw - zero_rwt
        idx_rwt = (np.ravel(zero_rwt.sum(axis=1)) > 0)
        zero_rwt = spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-2, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.qid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-2, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        zero_rwt = zero_rwt + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-2, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_cwt = (np.ravel(zero_rwt.sum(axis=1)) > 0)
        zero_rwt = spsp.lil_matrix(zero_rwt.shape)
        #zero_rwt[idx_cwt,idx_rwt] = 1
        zero_rwt[idx_rwt,idx_rwt] = 1

        # Pressure: T_iNN, T_NjN, T_NNk
        zero_p = zero_ru + zero_rv + zero_rw + zero_cu + zero_cv + zero_cw
        # Pressure: T_{N-2:N,N-2:N,N-2:N}
        zero_p = zero_p + spsp.kron(c1d.qid(res[1], res[1]-3, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-3, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-3, c1d.c1dbc.no_bc())))
        zero_p = zero_p + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-3, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-3, c1d.c1dbc.no_bc())))
        zero_p = zero_p + spsp.kron(c1d.qid(res[1], res[1]-3, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.qid(res[0], res[0]-3, c1d.c1dbc.no_bc())))
        zero_p = zero_p + spsp.kron(c1d.qid(res[1], res[1]-3, c1d.c1dbc.no_bc()), spsp.kron(c1d.qid(res[2],res[2]-3, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        # Pressure: T_000
        zero_p = zero_p + spsp.kron(c1d.sid(res[1], res[1]-1, c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2],res[2]-1, c1d.c1dbc.no_bc()), c1d.sid(res[0], res[0]-1, c1d.c1dbc.no_bc())))
        idx_p = (np.ravel(zero_p.sum(axis=1)) > 0)
        zero_p = spsp.lil_matrix(zero_p.shape)
        zero_p[idx_p,idx_p] = 1

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)
                mat[idx_ru,:] = 0
                mat[idx_cu,:] = 0
                mat[:,idx_ru] = 0
                mat[:,idx_cu] = 0
                mat = mat + zero_ru + zero_rut + zero_cu

            elif field_col == ("velocityy",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = c3d.i2j2k2d1(res[0], res[1], res[2], bc, -1.0).tolil()
                mat[idx_ru,:] = 0
                mat[idx_cu,:] = 0
                mat[:,idx_p] = 0

        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)
                mat[idx_rv,:] = 0
                mat[idx_cv,:] = 0
                mat[:,idx_rv] = 0
                mat[:,idx_cv] = 0
                mat = mat + zero_rv + zero_rvt + zero_cv

            elif field_col == ("velocityz",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = c3d.i2j2k2e1(res[0], res[1], res[2], bc, -1.0, yscale = yscale).tolil()
                mat[idx_rv,:] = 0
                mat[idx_cv,:] = 0
                mat[:,idx_p] = 0

        elif field_row == ("velocityz",""):
            if field_col == ("velocityx",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)
                mat[idx_rw,:] = 0
                mat[idx_cw,:] = 0
                mat[:,idx_rw] = 0
                mat[:,idx_cw] = 0
                mat = mat + zero_rw + zero_rwt + zero_cw

            elif field_col == ("temperature",""):
                mat = c3d.i2j2k2(res[0], res[1], res[2], bc, Ra/16.0).tolil()
                mat[idx_rw,:] = 0
                mat[idx_cw,:] = 0

            elif field_col == ("pressure",""):
                mat = c3d.i2j2k2f1(res[0], res[1], res[2], bc, -1.0, zscale = zscale).tolil()
                mat[idx_rw,:] = 0
                mat[idx_cw,:] = 0
                mat[:,idx_p] = 0

        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityy",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocityz",""):
                mat = c3d.i2j2k2(res[0], res[1], res[2], bc).tolil()
                mat[:,idx_rw] = 0
                mat[:,idx_cw] = 0

            elif field_col == ("temperature",""):
                mat = c3d.i2j2k2lapl(res[0], res[1], res[2], bc, yscale = yscale, zscale = zscale)

            elif field_col == ("pressure",""):
                mat = c3d.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

        elif field_row == ("pressure",""):
            if field_col == ("velocityx",""):
                mat = c3d.i1j1k1d1(res[0], res[1], res[2], bc).tolil()
                mat = mat + spsp.kron(c1d.sid(res[1],res[1]-1,c1d.c1dbc.no_bc()), spsp.kron(c1d.i1(res[2], c1d.c1dbc.no_bc()), c1d.i1d1(res[0], c1d.c1dbc.no_bc())))
                mat = mat + spsp.kron(c1d.i1(res[1],c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2], res[2]-1, c1d.c1dbc.no_bc()), c1d.i1d1(res[0], c1d.c1dbc.no_bc())))
                #mat[:,idx_cu] = 0
#                mat[idx_cu,:] = 0          
#                mat[idx_cv,:] = 0          
#                mat[idx_cw,:] = 0          
                mat[idx_p,:] = 0
#                mat = mat + zero_cu

            elif field_col == ("velocityy",""):
                mat = c3d.i1j1k1e1(res[0], res[1], res[2], bc).tolil()
                mat = mat + spsp.kron(c1d.i1d1(res[1],c1d.c1dbc.no_bc()), spsp.kron(c1d.sid(res[2], res[2]-1, c1d.c1dbc.no_bc()), c1d.i1(res[0], c1d.c1dbc.no_bc())))
                mat = mat + spsp.kron(c1d.i1d1(res[1],c1d.c1dbc.no_bc()), spsp.kron(c1d.i1(res[2], c1d.c1dbc.no_bc()), c1d.sid(res[0],res[0]-1, c1d.c1dbc.no_bc())))
                #mat[:,idx_cv] = 0
#                mat[idx_cu,:] = 0
#                mat[idx_cv,:] = 0
#                mat[idx_cw,:] = 0
                mat[idx_p,:] = 0
#                mat = mat + zero_cv

            elif field_col == ("velocityz",""):
                mat = c3d.i1j1k1f1(res[0], res[1], res[2], bc).tolil()
                mat = mat + spsp.kron(c1d.sid(res[1],res[1]-1,c1d.c1dbc.no_bc()), spsp.kron(c1d.i1d1(res[2], c1d.c1dbc.no_bc()), c1d.i1(res[0],c1d.c1dbc.no_bc())))
                mat = mat + spsp.kron(c1d.i1(res[1],c1d.c1dbc.no_bc()), spsp.kron(c1d.i1d1(res[2], c1d.c1dbc.no_bc()), c1d.sid(res[0],res[0]-1,c1d.c1dbc.no_bc())))
                #mat[:,idx_cw] = 0
#                mat[idx_cu,:] = 0
#                mat[idx_cv,:] = 0
#                mat[idx_cw,:] = 0
                mat[idx_p,:] = 0
#                mat = mat + zero_cw

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
