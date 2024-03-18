"""Module provides the functions to generate the Boussinesq rotating Rayleigh-Benard convection in a 3D box (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_3d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_3d import no_bc


class BoussinesqRRBCBoxVC(base_model.BaseModel):
    """Class to setup the Boussinesq rotating Rayleigh-Benard convection in a 3D box (velocity-continuity formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "taylor", "heating", "scale1d", "scale2d", "scale3d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocity", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature",""), ("pressure","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature",""), ("pressure","")]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            if field_row == ("temperature",""):
                fields = [("velocity","z")]
            else:
                fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]*res[1]*res[2]
        if self.use_galerkin:
            if field_row in [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature",""), ("pressure","")]:
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

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SINGLE

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

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
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'xz'}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'yx'}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:20}, 'y':{0:20}, 'z':{0:20}, 'priority':'zy'}

            # Stress-free/Stress-free/no-slip, Fixed flux/Fixed flux/Fixed temperature
            elif bcId == 4:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {'x':{0:20}, 'y':{0:21}, 'z':{0:20}, 'priority':'xz'}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:20}, 'priority':'yz'}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {'x':{0:21}, 'y':{0:21}, 'z':{0:20}, 'priority':'zsx'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:21}, 'y':{0:21}, 'z':{0:20}, 'priority':'zsy'}

            # Stress-free/Stress-free/no-slip, Fixed flux/Fixed flux/Fixed temperature
            elif bcId == 6:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {'x':{0:20}, 'y':{0:21}, 'z':{0:21}, 'priority':'xsz'}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {'x':{0:21}, 'y':{0:20}, 'z':{0:21}, 'priority':'ysz'}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {'x':{0:21}, 'y':{0:21}, 'z':{0:20}, 'priority':'zsx'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocity","x"):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","y"):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","z"):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 4:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'y':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocity","x"):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","y"):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","z"):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['y']['rt'] = 2
                    bc['z']['rt'] = 2

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the quasi-inverse operator"""

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","x") and field_col == field_row:
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, restriction = restriction)
            mat = utils.qid_from_idx(idx_u, np.prod(res))*mat

        elif field_row == ("velocity","y") and field_col == field_row:
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, restriction = restriction)
            mat = utils.qid_from_idx(idx_v, np.prod(res))*mat

        elif field_row == ("velocity","z") and field_col == field_row:
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, restriction = restriction)
            mat = utils.qid_from_idx(idx_w, np.prod(res))*mat

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']
        Ta = eq_params['taylor']
        T = Ta**0.5

        xscale = eq_params['scale1d']
        yscale = eq_params['scale2d']
        zscale = eq_params['scale3d']

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","x"):
            if field_col == ("velocity","x"):
                mat = geo.i2j2k2lapl(res[0], res[1], res[2], bc, xscale = xscale, yscale = yscale, zscale = zscale, restriction = restriction)
                mat = utils.qid_from_idx(idx_u, np.prod(res))*mat*utils.qid_from_idx(idx_u, np.prod(res))
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx_3d(idx_u, res[1], res[2], res[0], restriction = restriction)

            elif field_col == ("velocity","y"):
                mat = geo.i2j2k2(res[0], res[1], res[2], bc, T, restriction = restriction)
                mat = utils.qid_from_idx(idx_u, np.prod(res))*mat*utils.qid_from_idx(idx_v, np.prod(res))

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = geo.i2j2k2d1(res[0], res[1], res[2], bc, -1.0, xscale = xscale, restriction = restriction)
                mat = utils.qid_from_idx(idx_u, np.prod(res))*mat*utils.qid_from_idx(idx_p, np.prod(res))

        elif field_row == ("velocity","y"):
            if field_col == ("velocity","x"):
                mat = geo.i2j2k2(res[0], res[1], res[2], bc, -T, restriction = restriction)
                mat = utils.qid_from_idx(idx_v, np.prod(res))*mat*utils.qid_from_idx(idx_u, np.prod(res))

            elif field_col == ("velocity","y"):
                mat = geo.i2j2k2lapl(res[0], res[1], res[2], bc, xscale = xscale, yscale = yscale, zscale = zscale, restriction = restriction)
                mat = utils.qid_from_idx(idx_v, np.prod(res))*mat*utils.qid_from_idx(idx_v, np.prod(res))
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx_3d(idx_v, res[1], res[2], res[0], restriction = restriction)

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = geo.i2j2k2e1(res[0], res[1], res[2], bc, -1.0, yscale = yscale, restriction = restriction)
                mat = utils.qid_from_idx(idx_v, np.prod(res))*mat*utils.qid_from_idx(idx_p, np.prod(res))

        elif field_row == ("velocity","z"):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocity","z"):
                mat = geo.i2j2k2lapl(res[0], res[1], res[2], bc, xscale = xscale, yscale = yscale, zscale = zscale, restriction = restriction)
                mat = utils.qid_from_idx(idx_w, np.prod(res))*mat*utils.qid_from_idx(idx_w, np.prod(res))
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx_3d(idx_w, res[1], res[2], res[0], restriction = restriction)

            elif field_col == ("temperature",""):
                mat = geo.i2j2k2(res[0], res[1], res[2], bc, Ra, restriction = restriction)
                mat = utils.qid_from_idx(idx_w, np.prod(res))*mat

            elif field_col == ("pressure",""):
                mat = geo.i2j2k2f1(res[0], res[1], res[2], bc, -1.0, zscale = zscale, restriction = restriction)
                mat = utils.qid_from_idx(idx_w, np.prod(res))*mat*utils.qid_from_idx(idx_p, np.prod(res))

        elif field_row == ("temperature",""):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("velocity","z"):
                if self.linearize or bcs["bcType"] == self.FIELD_TO_RHS: 
                    if eq_params['heating'] == 0:
                        mat = geo.i2j2k2(res[0], res[1], res[2], bc, restriction = restriction)
                        mat = mat*utils.qid_from_idx(idx_w, np.prod(res))
                else:
                    mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2j2k2lapl(res[0], res[1], res[2], bc, xscale = xscale, yscale = yscale, zscale = zscale, restriction = restriction)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], res[1], res[2], 2, 2, 2, bc)

        elif field_row == ("pressure",""):
            if bcs["bcType"] == self.SOLVER_NO_TAU:
                mat = geo.zblk(res[0], res[1], res[2], 1, 1, 1, no_bc())
            else:
                if field_col == ("velocity","x"):
                    bc['x']['cr'] = 1
                    bc['x']['rt'] = 1
                    bc['y']['cr'] = 1
                    bc['y']['rt'] = 1
                    bc['z']['cr'] = 1
                    bc['z']['rt'] = 1
                    mat = geo.i1j1k1d1(res[0]+1, res[1]+1, res[2]+1, bc, xscale = xscale, restriction = restriction)
                    mat = utils.qid_from_idx(idx_p, np.prod(res))*mat*utils.qid_from_idx(idx_u, np.prod(res))

                elif field_col == ("velocity","y"):
                    bc['x']['cr'] = 1
                    bc['x']['rt'] = 1
                    bc['y']['cr'] = 1
                    bc['y']['rt'] = 1
                    bc['z']['cr'] = 1
                    bc['z']['rt'] = 1
                    mat = geo.i1j1k1e1(res[0]+1, res[1]+1, res[2]+1, bc, yscale = yscale, restriction = restriction)
                    mat = utils.qid_from_idx(idx_p, np.prod(res))*mat*utils.qid_from_idx(idx_v, np.prod(res))

                elif field_col == ("velocity","z"):
                    bc['x']['cr'] = 1
                    bc['x']['rt'] = 1
                    bc['y']['cr'] = 1
                    bc['y']['rt'] = 1
                    bc['z']['cr'] = 1
                    bc['z']['rt'] = 1
                    mat = geo.i1j1k1f1(res[0]+1, res[1]+1, res[2]+1, bc, zscale = zscale, restriction = restriction)
                    mat = utils.qid_from_idx(idx_p, np.prod(res))*mat*utils.qid_from_idx(idx_w, np.prod(res))

                elif field_col == ("temperature",""):
                    mat = geo.zblk(res[0], res[1], res[2], 1, 1, 1, bc)

                elif field_col == ("pressure",""):
                    mat = geo.zblk(res[0], res[1], res[2], 1, 1, 1, bc)
                    mat = mat + utils.id_from_idx_3d(idx_p, res[1], res[2], res[0], restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocity","x"):
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, 1.0/Pr, restriction = restriction)
            S = utils.qid_from_idx(idx_u, np.prod(res))
            mat = S*mat*S

        elif field_row == ("velocity","y"):
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, 1.0/Pr, restriction = restriction)
            S = utils.qid_from_idx(idx_v, np.prod(res))
            mat = S*mat*S

        elif field_row == ("velocity","z"):
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, 1.0/Pr, restriction = restriction)
            S = utils.qid_from_idx(idx_w, np.prod(res))
            mat = S*mat*S

        elif field_row == ("temperature",""):
            mat = geo.i2j2k2(res[0], res[1], res[2], bc, restriction = restriction)

        elif field_row == ("pressure",""):
            mat = geo.zblk(res[0], res[1], res[2], 1, 1, 1, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def zero_blocks(self, res, eigs):
        """Build restriction matrices"""
    
        # U:
        idx_u = utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], res[1]-1), utils.qidx(res[2], 0), utils.qidx(res[0], 0))
        idx_u = np.union1d(idx_u, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], res[2]-1), utils.qidx(res[0], 0)))

        # V:
        idx_v = utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-1))
        idx_v = np.union1d(idx_v, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], res[2]-1), utils.qidx(res[0], 0)))

        # W:
        idx_w = utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-1))
        idx_w = np.union1d(idx_w,  utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], res[1]-1), utils.qidx(res[2], 0), utils.qidx(res[0], 0)))

        # Pressure: T_iNN, T_NjN, T_NNk
        idx_p = utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-1))
        idx_p = np.union1d(idx_p, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], res[1]-1), utils.qidx(res[2], 0), utils.qidx(res[0], 0)))
        idx_p = np.union1d(idx_p, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], res[2]-1), utils.qidx(res[0], 0)))
        # Pressure: T_{N-2:N,N-2:N,N-2:N}
        idx_p = np.union1d(idx_p, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], 0), utils.qidx(res[2], res[2]-3), utils.qidx(res[0], res[0]-3)))
        idx_p = np.union1d(idx_p, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], res[1]-3), utils.qidx(res[2], res[2]-3), utils.qidx(res[0], 0)))
        idx_p = np.union1d(idx_p, utils.idx_kron_3d(res[1], res[2], res[0], utils.qidx(res[1], res[1]-3), utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-3)))
        # Pressure: T_000
        idx_p = np.union1d(idx_p, utils.idx_kron_3d(res[1], res[2], res[0], utils.sidx(res[1], res[1]-1), utils.sidx(res[2], res[2]-1), utils.sidx(res[0], res[0]-1)))

        return (idx_u, idx_v, idx_w, idx_p)
