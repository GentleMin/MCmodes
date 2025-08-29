"""Module provides the functions to generate the 2D Boussinesq Rayleigh-Benard convection in a ring (velocity-continuity formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cylindrical.annulus_radius as annulus
import quicc.base.base_model as base_model
from quicc.geometry.cylindrical.annulus_radius_boundary import no_bc


class BoussinesqRBRingVC(base_model.BaseModel):
    """Class to setup the 2D Boussinesq Rayleigh-Benard convection in a ring (velocity-continuity formulation)"""

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "ro", "rratio"]

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocityx", "velocityy", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocityx",""), ("velocityy",""), ("temperature",""), ("pressure","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocityx",""), ("velocityy",""), ("temperature",""), ("pressure","")]

        return fields

    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        fields = []

        return fields

    def block_size(self, res, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("velocityx","") or field_row == ("velocityy","") or field_row == ("temperature",""):
                shift_r = 2
            else:
                shift_r = 0

            gal_n = (res[0] - shift_r)

        else:
            gal_n = tau_n
            shift_r = 0

        block_info = (tau_n, gal_n, (shift_r,0,0), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex
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
            a, b = annulus.linear_r2x(eq_params['ro'], eq_params['rratio'])

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            # No-slip/No-slip, Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("velocityy",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-20, 'r':0}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {0:20}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {0:20}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {0:20}

            # Stress-free/Stress-free, Fixed flux/Fixed flux
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("velocityy",""):
                        bc = {0:-24, 'r':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-21, 'r':0}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {0:20}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {0:24, 'c':{'a':a, 'b':b}}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {0:21}
            
            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 2:
                if self.use_galerkin:
                    if field_col == ("velocityx",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("velocityy",""):
                        bc = {0:-24, 'r':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-21, 'r':0}

                else:
                    if field_row == ("velocityx","") and field_col == ("velocityx",""):
                        bc = {0:20}
                    elif field_row == ("velocityy","") and field_col == ("velocityy",""):
                        bc = {0:24, 'c':{'a':a, 'b':b}}
                    elif field_row == ("temperature","") and field_col == ("temperature",""):
                        bc = {0:21}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['r'] = 2
                elif field_row == ("temperature",""):
                    bc['r'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocityx",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("velocityy",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-20, 'r':0}

                elif bcId == 1:
                    if field_col == ("velocityx",""):
                        bc = {0:-20, 'r':0}
                    elif field_col == ("velocityy",""):
                        bc = {0:-24, 'r':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-21, 'r':0}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocityx",""):
                    bc['r'] = 2
                elif field_row == ("velocityy",""):
                    bc['r'] = 2
                elif field_row == ("temperature",""):
                    bc['r'] = 2

        return bc

    def stencil(self, res, eq_params, eigs, bcs, field_row):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return annulus.stencil(res[0], bc)

    def qi(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create the quasi-inverse operator"""

        a, b = annulus.linear_r2x(eq_params['ro'], eq_params['rratio'])

        idx_u, idx_v, idx_p = self.zero_blocks(res, eigs)

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = annulus.i2x2(res[0], a, b, bc)
            mat = utils.qid_from_idx(idx_u, res[0])*mat

        elif field_row == ("velocityy",""):
            mat = annulus.i2x2(res[0], a, b, bc)
            mat = utils.qid_from_idx(idx_v, res[0])*mat

        elif field_row == ("temperature",""):
            mat = annulus.i2x2(res[0], a, b, bc)

        elif field_row == ("pressure",""):
            mat = annulus.zblk(res[0], bc)

        return mat

    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']
        m = eigs[0]

        a, b = annulus.linear_r2x(eq_params['ro'], eq_params['rratio'])

        idx_u, idx_v, idx_p = self.zero_blocks(res, eigs)

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocityx",""):
            if field_col == ("velocityx",""):
                mat = annulus.i2x2laplh(res[0], m, a, b, bc)
                bc[0] = min(bc[0], 0)
                mat = mat + annulus.i2(res[0], a, b, bc, -1.0)
                mat = utils.qid_from_idx(idx_u, res[0])*mat*utils.qid_from_idx(idx_u, res[0])
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx(idx_u, res[0])

            elif field_col == ("velocityy",""):
                mat = annulus.i2(res[0], a, b, bc, -2.0*1j*m)
                mat = utils.qid_from_idx(idx_u, res[0])*mat*utils.qid_from_idx(idx_v, res[0])

            elif field_col == ("temperature",""):
                mat = annulus.zblk(res[0], bc)
                #mat = annulus.i2x2(res[0], a, b, bc, Ra)
                #mat = utils.qid_from_idx(idx_u, res[0])*mat

            elif field_col == ("pressure",""):
                mat = annulus.i2x2d1(res[0], a, b, bc, -1.0)
                mat = utils.qid_from_idx(idx_u, res[0])*mat*utils.qid_from_idx(idx_p, res[0])

        elif field_row == ("velocityy",""):
            if field_col == ("velocityx",""):
                mat = annulus.i2(res[0], a, b, bc, 2.0*1j*m)
                mat = utils.qid_from_idx(idx_v, res[0])*mat*utils.qid_from_idx(idx_u, res[0])

            elif field_col == ("velocityy",""):
                mat = annulus.i2x2laplh(res[0], m, a, b, bc)
                bc[0] = min(bc[0], 0)
                mat = mat + annulus.i2(res[0], a, b, bc, -1.0)
                mat = utils.qid_from_idx(idx_v, res[0])*mat*utils.qid_from_idx(idx_v, res[0])
                if bcs["bcType"] == self.SOLVER_HAS_BC:
                    mat = mat + utils.id_from_idx(idx_v, res[0])

            elif field_col == ("temperature",""):
                mat = annulus.zblk(res[0], bc)

            elif field_col == ("pressure",""):
                mat = annulus.i2x1(res[0], a, b, bc, -1j*m)
                mat = utils.qid_from_idx(idx_v, res[0])*mat*utils.qid_from_idx(idx_p, res[0])

        elif field_row == ("temperature",""):
            if field_col == ("velocityx",""):
                if self.linearize:
                    mat = annulus.i2(res[0], a, b, bc)
                    mat = mat*utils.qid_from_idx(idx_u, res[0])
                else:
                    mat = annulus.zblk(res[0], bc)

            elif field_col == ("velocityy",""):
                mat = annulus.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = annulus.i2x2laplh(res[0], m, a, b, bc)

            elif field_col == ("pressure",""):
                mat = annulus.zblk(res[0], bc)

        elif field_row == ("pressure",""):
            if bcs["bcType"] == self.SOLVER_HAS_BC:
                if field_col == ("velocityx",""):
                    bc['cr'] = 1
                    bc['rt'] = 1
                    #bc['zb'] = 1
                    mat = annulus.i1x1div(res[0]+1, a, b, bc)
                    mat = utils.qid_from_idx(idx_p, res[0])*mat*utils.qid_from_idx(idx_u, res[0])

                elif field_col == ("velocityy",""):
                    bc['cr'] = 1
                    bc['rt'] = 1
                    #bc['zb'] = 1
                    mat = annulus.i1(res[0]+1, a, b, bc, 1j*m)
                    mat = utils.qid_from_idx(idx_p, res[0])*mat*utils.qid_from_idx(idx_v, res[0])

                elif field_col == ("temperature",""):
                    mat = annulus.zblk(res[0], bc)

                elif field_col == ("pressure",""):
                    mat = annulus.zblk(res[0], bc)
                    mat = mat + utils.id_from_idx(idx_p, res[0])
            else:
                mat = annulus.zblk(res[0], no_bc())

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']

        a, b = annulus.linear_r2x(eq_params['ro'], eq_params['rratio'])
        m = eigs[0]

        idx_u, idx_v, idx_p = self.zero_blocks(res, eigs)

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocityx",""):
            mat = annulus.i2x2(res[0], a, b, bc, 1.0/Pr)
            S = utils.qid_from_idx(idx_u, res[0])
            mat = S*mat*S

        elif field_row == ("velocityy",""):
            mat = annulus.i2x2(res[0], a, b, bc, 1.0/Pr)
            S = utils.qid_from_idx(idx_v, res[0])
            mat = S*mat*S

        elif field_row == ("temperature",""):
            mat = annulus.i2x2(res[0], a, b, bc)

        elif field_row == ("pressure",""):
            mat = annulus.zblk(res[0], bc)

        return mat

    def zero_blocks(self, res, eigs, restriction = None):
        """Build restriction matrices"""

        # U: T_iN, T_Ni
        idx_u = []
#        idx_u = utils.qidx(res[0], res[0]-1)

        # V: T_iN, T_Ni
        idx_v = []
#        idx_v = utils.qidx(res[0], res[0]-1)

        # Pressure: T_iN, T_Nk
        idx_p = []
#        idx_p = utils.qidx(res[0], res[0]-2)
        # Pressure: T_00
        if eigs[0] == 0:
            idx_p = utils.sidx(res[0], res[0]-1)

        return (idx_u, idx_v, idx_p)
