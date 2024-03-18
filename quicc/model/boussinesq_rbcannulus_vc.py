"""Module provides the functions to generate the Boussinesq Rayleigh-Benard convection in a cylindrical annulus modified for parity (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cylindrical.annulus as geo
import quicc.base.base_model as base_model
from quicc.geometry.cylindrical.annulus_boundary import no_bc


class BoussinesqRBCAnnulusVC(base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a cylindrical annulus modified for parity (velocity-continuity formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "ro", "rratio", "scale3d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocity", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocity","r"), ("velocity","theta"), ("velocity","z"), ("temperature",""), ("pressure","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocity","r"), ("velocity","theta"), ("velocity","z"), ("temperature",""), ("pressure","")]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("velocity","r"), ("velocity","theta"), ("velocity","z"), ("temperature","")]:
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
            if field_row == ("velocity","r") or field_row == ("velocity","theta") or field_row == ("velocity","z") or field_row == ("temperature",""):
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

        # Matrix operator is complex
        is_complex = True

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_SINGLE_RHS

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        m = int(eigs[0])
        a, b = geo.rad.linear_r2x(eq_params['ro'], eq_params['rratio'])

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
                    if field_col == ("velocity","r"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","theta"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","r") and field_col == ("velocity","r"):
                        bc = {'r':{0:20}, 'z':{0:20}, 'priority':'r'}
                    elif field_row == ("velocity","theta") and field_col == ("velocity","theta"):
                        bc = {'r':{0:20}, 'z':{0:20}, 'priority':'r'}
                    elif field_row == ("velocity","z") and field_col == ("velocity","z"):
                        bc = {'r':{0:20}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'r':{0:20}, 'z':{0:20}, 'priority':'r'}

            # Stress-free/Stress-free, Fixed flux/Fixed flux
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocity","r"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","theta"):
                        bc = {'r':{0:-24, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'r':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}

                else:
                    if field_row == ("velocity","r") and field_col == ("velocity","r"):
                        bc = {'r':{0:20}, 'z':{0:21}, 'priority':'r'}
                    elif field_row == ("velocity","theta") and field_col == ("velocity","theta"):
                        bc = {'r':{0:24, 'c':{'a':a, 'b':b}}, 'z':{0:21}, 'priority':'sr'}
                    elif field_row == ("velocity","z") and field_col == ("velocity","z"):
                        bc = {'r':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'r':{0:21}, 'z':{0:21}, 'priority':'sr'}
            
            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("velocity","r"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","theta"):
                        bc = {'r':{0:-24, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'r':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-21, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","r") and field_col == ("velocity","r"):
                        bc = {'r':{0:20}, 'z':{0:20}, 'priority':'r'}
                    elif field_row == ("velocity","theta") and field_col == ("velocity","theta"):
                        bc = {'r':{0:24, 'c':{'a':a, 'b':b}}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("velocity","z") and field_col == ("velocity","z"):
                        bc = {'r':{0:21}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {'r':{0:21}, 'z':{0:20}, 'priority':'z'}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocity","r"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","theta"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","z"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocity","r"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","theta"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-20, 'rt':0}}

                elif bcId == 1:
                    if field_col == ("velocity","r"):
                        bc = {'r':{0:-20, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("velocity","theta"):
                        bc = {'r':{0:-24, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","z"):
                        bc = {'r':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-21, 'rt':0}, 'z':{0:-21, 'rt':0}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocity","r"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","theta"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","z"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2
                elif field_row == ("temperature",""):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 2

        return bc

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        a, b = geo.rad.linear_r2x(eq_params['ro'], eq_params['rratio'])

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","r") and field_col == field_row:
            mat = geo.i2j2r3(res[0], res[2], a, b, bc)
            mat = utils.qid_from_idx(idx_u, res[0]*res[2])*mat

        elif field_row == ("velocity","theta") and field_col == field_row:
            mat = geo.i2j2r3(res[0], res[2], a, b, bc)
            mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat

        elif field_row == ("velocity","z") and field_col == field_row:
            mat = geo.i2j2r2(res[0], res[2], a, b, bc)
            mat = utils.qid_from_idx(idx_w, res[0]*res[2])*mat

        elif field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2r2(res[0], res[2], a, b, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        assert(eigs[0].is_integer())

        Ra = eq_params['rayleigh']
        zscale = eq_params['scale3d']

        m = int(eigs[0])
        a, b = geo.rad.linear_r2x(eq_params['ro'], eq_params['rratio'])

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","r"):
            if field_col == ("velocity","r"):
                mat = geo.i2j2r3vlaplr_1(res[0], res[2], m, a, b, bc, zscale = zscale)
                mat = utils.qid_from_idx(idx_u, res[0]*res[2])*mat*utils.qid_from_idx(idx_u, res[0]*res[2])
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx_2d(idx_u, res[2], res[0])

            elif field_col == ("velocity","theta"):
                mat = geo.i2j2(res[0], res[2], a, b, bc, -2.0*1j*m)
                mat = utils.qid_from_idx(idx_u, res[0]*res[2])*mat*utils.qid_from_idx(idx_v, res[0]*res[2])

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = geo.i2j2r3d1r_2(res[0], res[2], a, b, bc, -1.0)
                mat = utils.qid_from_idx(idx_u, res[0]*res[2])*mat*utils.qid_from_idx(idx_p, res[0]*res[2])

        elif field_row == ("velocity","theta"):
            if field_col == ("velocity","r"):
                mat = geo.i2j2(res[0], res[2], a, b, bc, 2.0*1j*m)
                mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat*utils.qid_from_idx(idx_u, res[0]*res[2])

            elif field_col == ("velocity","theta"):
                mat = geo.i2j2r3vlaplr_1(res[0], res[2], m, a, b, bc, zscale = zscale)
                mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat*utils.qid_from_idx(idx_v, res[0]*res[2])
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx_2d(idx_v, res[2], res[0])

            elif field_col == ("velocity","z"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("pressure",""):
                mat = geo.i2j2(res[0], res[2], a, b, bc, -1j*m)
                mat = utils.qid_from_idx(idx_v, res[0]*res[2])*mat*utils.qid_from_idx(idx_p, res[0]*res[2])

        elif field_row == ("velocity","z"):
            if field_col == ("velocity","r"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","theta"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","z"):
                mat = geo.i2j2r2lapl(res[0], res[2], m, a, b, bc, zscale = zscale)
                mat = utils.qid_from_idx(idx_w, res[0]*res[2])*mat*utils.qid_from_idx(idx_w, res[0]*res[2])
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx_2d(idx_w, res[2], res[0])

            elif field_col == ("temperature",""):
                mat = geo.i2j2r2(res[0], res[2], a, b, bc, Ra)
                mat = utils.qid_from_idx(idx_w, res[0]*res[2])*mat

            elif field_col == ("pressure",""):
                mat = geo.i2j2e1(res[0], res[2], a, b, bc, -1.0, zscale = zscale)
                mat = utils.qid_from_idx(idx_w, res[0]*res[2])*mat*utils.qid_from_idx(idx_p, res[0]*res[2])

        elif field_row == ("temperature",""):
            if field_col == ("velocity","r"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","theta"):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("velocity","z"):
                if self.linearize:
                    mat = geo.i2j2r2(res[0], res[2], a, b, bc)
                    mat = mat*utils.qid_from_idx(idx_w, res[0]*res[2])
                else:
                    mat = geo.zblk(res[0], res[2], 2, 2, bc)

            elif field_col == ("temperature",""):
                mat = geo.i2j2r2lapl(res[0], res[2], m, a, b, bc, zscale = zscale)

            elif field_col == ("pressure",""):
                mat = geo.zblk(res[0], res[2], 2, 2, bc)

        elif field_row == ("pressure",""):
            if bcs["bcType"] == self.SOLVER_HAS_BC:
                if field_col == ("velocity","r"):
                    bc['r']['cr'] = 1
                    bc['r']['rt'] = 1
                    #bc['r']['zb'] = 1
                    bc['z']['cr'] = 1
                    bc['z']['rt'] = 1
                    bc['z']['zb'] = 1
                    mat = geo.i1j1r1d1(res[0]+1, res[2]+1, a, b, bc)
                    mat = utils.qid_from_idx(idx_p, res[0]*res[2])*mat*utils.qid_from_idx(idx_u, res[0]*res[2])

                elif field_col == ("velocity","theta"):
                    bc['r']['cr'] = 1
                    bc['r']['rt'] = 1
                    #bc['r']['zb'] = 1
                    bc['z']['cr'] = 1
                    bc['z']['rt'] = 1
                    bc['z']['zb'] = 1
                    mat = geo.i1j1(res[0]+1, res[2]+1, a, b, bc, 1j*m)
                    mat = utils.qid_from_idx(idx_p, res[0]*res[2])*mat*utils.qid_from_idx(idx_v, res[0]*res[2])

                elif field_col == ("velocity","z"):
                    bc['r']['cr'] = 1
                    bc['r']['rt'] = 1
                    #bc['r']['zb'] = 1
                    bc['z']['cr'] = 1
                    bc['z']['rt'] = 1
                    bc['z']['zb'] = 1
                    mat = geo.i1j1r2e1(res[0]+1, res[2]+1, a, b, bc, zscale = zscale)
                    mat = utils.qid_from_idx(idx_p, res[0]*res[2])*mat*utils.qid_from_idx(idx_w, res[0]*res[2])

                elif field_col == ("temperature",""):
                    mat = geo.zblk(res[0], res[2], 1, 1, bc)

                elif field_col == ("pressure",""):
                    mat = geo.zblk(res[0], res[2], 1, 1, bc)
                    mat = mat + utils.id_from_idx_2d(idx_p, res[2], res[0])
            else:
                mat = geo.zblk(res[0], res[2], 1, 1, no_bc())

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        assert(eigs[0].is_integer())

        Pr = eq_params['prandtl']

        m = int(eigs[0])
        a, b = geo.rad.linear_r2x(eq_params['ro'], eq_params['rratio'])

        idx_u, idx_v, idx_w, idx_p = self.zero_blocks(res, eigs)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocity","r"):
            mat = geo.i2j2r2(res[0], res[2], a, b, bc, 1.0/Pr)
            S = utils.qid_from_idx(idx_u, res[0]*res[2])
            mat = S*mat*S

        elif field_row == ("velocity","theta"):
            mat = geo.i2j2r2(res[0], res[2], a, b, bc, 1.0/Pr)
            S = utils.qid_from_idx(idx_v, res[0]*res[2])
            mat = S*mat*S

        elif field_row == ("velocity","z"):
            mat = geo.i2j2r2(res[0], res[2], a, b, bc, 1.0/Pr)
            S = utils.qid_from_idx(idx_w, res[0]*res[2])
            mat = S*mat*S

        elif field_row == ("temperature",""):
            mat = geo.i2j2r2(res[0], res[2], a, b, bc)

        elif field_row == ("pressure",""):
            mat = geo.zblk(res[0], res[2], 1, 1, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def zero_blocks(self, res, eigs):
        """Build restriction matrices"""

        assert(eigs[0].is_integer())

        m = int(eigs[0])

        # U: T_iN, T_Ni
        idx_u = utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-1), utils.qidx(res[0], 0))
#        idx_u = np.union1d(idx_u, utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-1)))

        # V: T_iN, T_Ni
        idx_v = utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-1), utils.qidx(res[0], 0))
#        idx_v = np.union1d(idx_v, utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-1)))

        # W: T_Nk, T_N-1K
        idx_w = utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-2))

        # Pressure: T_iN, T_Nk
        idx_p = utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-1), utils.qidx(res[0], 0))
        idx_p = np.union1d(idx_p, utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], res[2]-3), utils.qidx(res[0], res[0]-4)))
        # Pressure: T_00
        if m == 0:
            idx_p = np.union1d(idx_p, utils.idx_kron_2d(res[2], res[0], utils.qidx(res[2], 0), utils.qidx(res[0], res[0]-1)))
            idx_p = np.union1d(idx_p, utils.idx_kron_2d(res[2], res[0], utils.sidx(res[2], res[2]-1), utils.sidx(res[0], res[0]-1)))

        return (idx_u, idx_v, idx_w, idx_p)
