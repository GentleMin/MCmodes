"""Module provides the functions to generate the Boussinesq rotating Rayleigh-Benard convection in a infinite duct (1 periodic direction) (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_2d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_2d import no_bc


class BoussinesqRRBCDuctVC(base_model.BaseModel):
    """Class to setup the Boussinesq rotating Rayleigh-Benard convection in a infinite duct (1 periodic direction) (velocity-continuity formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "taylor", "heating", "scale1d", "scale3d"]

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

        tau_n = res[0]*res[2]
        if self.use_galerkin:
            if field_row in [("velocity","x"), ("velocity","y"), ("velocity","z"), ("temperature",""), ("pressure","")]:
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

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_SINGLE_RHS

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
            # No-slip/No-slip, Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'z'}

            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:20}, 'priority':'x'}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}

            # Stress-free/Stress-free, Fixed flux/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","x") and field_col == field_row:
                        bc = {'x':{0:20}, 'z':{0:21}, 'priority':'sx'}
                    elif field_row == ("velocity","y") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:21}, 'priority':'sz'}
                    elif field_row == ("velocity","z") and field_col == field_row:
                        bc = {'x':{0:21}, 'z':{0:20}, 'priority':'z'}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocity","x"):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","y"):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","z"):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 1:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}

                elif bcId == 2:
                    if field_col == ("velocity","x"):
                        bc = {'x':{0:-20, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","y"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'x':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocity","x"):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","y"):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","z"):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['x']['rt'] = 2
                    bc['z']['rt'] = 2

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("temperature","") and field_col == ("velocity","z"):
            if eq_params['heating'] == 0:
                mat = geo.i2j2(res[0], res[2], bc, -1.0)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        idx_v, idx_lp, idx_rp = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","x") and field_col == field_row:
            mat = geo.i2j2(res[0], res[2], bc)

        elif field_row == ("velocity","y") and field_col == field_row:
            mat = geo.i2j2(res[0], res[2], bc)
            if eigs[0] == 0 and bc['x'][0] == 21 and bc['z'][0] == 21:
                mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat

        elif field_row == ("velocity","z") and field_col == field_row:
            mat = geo.i2j2(res[0], res[2], bc)

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2(res[0], res[2], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']
        Ta = eq_params['taylor']
        T = Ta**0.5

        xscale = eq_params['scale1d']
        zscale = eq_params['scale3d']

        k = eigs[0]

        idx_v, idx_lp, idx_rp = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","x"):
            if field_col == ("velocity","x"):
                mat = geo.i2j2lapl(res[0], res[2], k, bc, xscale = xscale, zscale = zscale)

            elif field_col == ("velocity","y"):
                mat = geo.i2j2(res[0], res[2], bc, T)

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = geo.i2j2d1(res[0], res[2], bc, -1.0, xscale = xscale)*utils.qid_from_idx(idx_rp, res[0]*res[2])

        elif field_row == ("velocity","y"):
            if field_col == ("velocity","x"):
                mat = geo.i2j2(res[0], res[2], bc, -T)
                if k == 0 and bc['x'][0] == 21 and bc['z'][0] == 21:
                    mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat

            elif field_col == ("velocity","y"):
                mat = geo.i2j2lapl(res[0], res[2], k, bc, xscale = xscale, zscale = zscale)
                if k == 0 and bc['x'][0] == 21 and bc['z'][0] == 21:
                    mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat
                    if bcs["bcType"] == self.SOLVER_HAS_BC:
                        tmp = spsp.lil_matrix(mat.shape)
                        tmp[-2*res[0]+1, 0] = 1
                        mat = mat + tmp

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = geo.i2j2(res[0], res[2], bc, -1j*k)*utils.qid_from_idx(idx_rp, res[0]*res[2])
                if k == 0 and bc['x'][0] == 21 and bc['z'][0] == 21:
                    mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat

        elif field_row == ("velocity","z"):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","z"):
                mat = geo.i2j2lapl(res[0], res[2], k, bc, xscale = xscale, zscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.i2j2(res[0], res[2], bc, Ra)

            elif field_col == ("pressure",""):
                mat = geo.i2j2e1(res[0], res[2], bc, -1.0, zscale = zscale)*utils.qid_from_idx(idx_rp, res[0]*res[2])

        elif field_row == ("temperature",""):
            if field_col == ("velocity","x"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","y"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","z"):
                if self.linearize or bcs["bcType"] == self.FIELD_TO_RHS:
                    if eq_params['heating'] == 0:
                        mat = geo.i2j2(res[0], res[2], bc)
                else:
                    mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2j2lapl(res[0], res[2], k, bc, xscale = xscale, zscale = zscale)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

        elif field_row == ("pressure",""):
            if bcs["bcType"] == self.SOLVER_HAS_BC:
                if field_col == ("velocity","x"):
                    mat = geo.i1j1d1(res[0]+1, res[2]+1, bc, xscale = xscale)
                    mat = utils.qid_from_idx(idx_p, res[0]*res[2])*mat

                elif field_col == ("velocity","y"):
                    mat = geo.i1j1(res[0]+1, res[2]+1, bc, 1j*k)
                    mat = utils.qid_from_idx(idx_p, res[0]*res[2])*mat

                elif field_col == ("velocity","z"):
                    mat = geo.i1j1e1(res[0]+1, res[2]+1, bc, zscale = zscale)
                    mat = utils.qid_from_idx(idx_p, res[0]*res[2])*mat

                elif field_col == ("temperature",""):
                    mat = geo.zblk(res[0], res[2], 1, 1, bc)

                elif field_col == ("pressure",""):
                    mat = geo.zblk(res[0], res[2], 1, 1, bc)
                    zero_p = spsp.lil_matrix(mat.shape)
                    zero_p[-2*res[0], -res[0]-2] = 1
                    zero_p[-2*res[0]+1, -res[0]-1] = 1
                    zero_p[-res[0], -2] = 1
                    zero_p[-res[0]+1, -1] = 1
                    if k == 0:
                        zero_p[0, 0] = 1
                        zero_p[1, res[0]-1] = 1
                        zero_p[res[0], -res[0]] = 1
                        zero_p[res[0]+1, -3] = 1
                    mat = mat + zero_p
            else:
                mat = geo.zblk(res[0], res[2], 1, 1, no_bc())

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']

        idx_v, idx_lp, idx_rp = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocity","x"):
            mat = geo.i2j2(res[0], res[2], bc, 1.0/Pr)

        elif field_row == ("velocity","y"):
            mat = geo.i2j2(res[0], res[2], bc, 1.0/Pr)
            if eigs[0] == 0 and bc['x'][0] == 21 and bc['z'][0] == 21:
                S = utils.qid_from_idx(idx_v, res[0]*res[2])
                mat = S*mat*S

        elif field_row == ("velocity","z"):
            mat = geo.i2j2(res[0], res[2], bc, 1.0/Pr)

        elif field_row == ("temperature",""):
            mat = geo.i2j2(res[0], res[2], bc)

        elif field_row == ("pressure",""):
            mat = geo.zblk(res[0], res[2], 1, 1, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def zero_blocks(self, res, eigs):
        """Build restriction matrices"""

        idx_v = np.array([])
        idx_lp = utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-2), utils.sidx(res[0], res[0]-2))
        idx_rp = utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-2), utils.qidx(res[0], res[0]-2))
        if eigs[0] == 0:        
            idx_lp = np.union1d(idx_lp, utils.idx_kron_2d(res[2], res[0], utils.sidx(res[2], res[2]-2), utils.sidx(res[0], res[0]-2)))
            idx_rp = np.union1d(idx_rp, utils.idx_kron_2d(res[2], res[0], utils.sidx(res[2], res[2]-1), utils.sidx(res[0], res[0]-1)))
            idx_rp = np.union1d(idx_rp, utils.idx_kron_2d(res[2], res[0], utils.sidx(res[2], res[2]-1), utils.qidx(res[0], res[0]-1)))
            idx_rp = np.union1d(idx_rp, utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-1), utils.sidx(res[0], res[0]-1)))
            idx_rp = np.union1d(idx_rp, utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-1), utils.qidx(res[0], res[0]-3)))
        if eigs[0] == 0:        
            idx_v = utils.idx_kron_2d(res[2], res[0], np.array([res[0]-2]), np.array([1]))

        return (idx_v, idx_lp, idx_rp)
