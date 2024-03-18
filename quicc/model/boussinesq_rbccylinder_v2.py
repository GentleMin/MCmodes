"""Module provides the functions to generate the Boussinesq Rayleigh-Benard convection in a cylinder (Toroidal/Poloidal formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import functools

import quicc.base.utils as utils
import quicc.geometry.cylindrical.cylinder_worland as geo
import quicc.base.base_model as base_model
from quicc.geometry.cylindrical.cylinder_boundary_worland import no_bc


class BoussinesqRBCCylinderConfig:
    """Class to setup the Boussinesq Rayleigh-Benard convection in a cylinder (Toroidal/Poloidal formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "gamma", "scale3d"]

    def automatic_parameters(self, eq_params):
        """Extend parameters with automatically computable values"""

        # Merge aspect ratio into scale factor
        d = {"scale3d":eq_params["scale3d"]*eq_params["gamma"]}

        return d

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocity", "temperature"]

    def stencil(self, res, eq_params, eigs, bcs, field_row, make_square):
        """Create the galerkin stencil"""
        
        assert(eigs[0].is_integer())

        m = int(eigs[0])

        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return geo.stencil(res[0], res[2], m, bc, make_square)

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = True

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_SINGLE_RHS

        return self.compile_equation_info(res, field_row, is_complex, index_mode)


class BoussinesqRBCCylinder(BoussinesqRBCCylinderConfig, base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a cylinder (Toroidal/Poloidal formulation)"""

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocity","tor"), ("velocity","pol"), ("magnetic","tor"), ("magnetic","pol"), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  [("velocity","tor"), ("velocity","pol"), ("temperature","")]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            if field_row in [("velocity","pol")]:
                fields = [("temperature","")]
            else:
                fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("velocity","tor"),("velocity","pol"),("temperature","")]:
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
            if field_row == ("velocity","tor"):
                shift_r = 1
                shift_z = 2
            elif field_row == ("velocity","pol"):
                shift_r = 2
                shift_z = 4
            elif field_row == ("temperature",""):
                shift_r = 1
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

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        m = eigs[0]
        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            Ra = eq_params['rayleigh']
            zscale = eq_params['scale3d']

            bc = no_bc()
            if field_col[0] in ['magnetic']:
                bcId = bcs.get('velocity', -1)
            else:
                bcId = bcs.get(field_col[0], -1)
            # No-slip/No-slip, Fixed temperature/Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocity","tor"):
                        if m == 0:
                            bc = {'r':{0:-10, 'rt':0}, 'z':{0:-20, 'rt':0}}
                        else:
                            bc = {'r':{0:-11, 'rt':0}, 'z':{0:-20, 'rt':0}}
                    elif field_col == ("velocity","pol"):
                        if m == 0:
                            bc = {'r':{0:-20, 'rt':0}, 'z':{0:-40, 'rt':0}}
                        else:
                            bc = {'r':{0:-22, 'rt':0}, 'z':{0:-40, 'rt':0}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-10, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("velocity","tor") and field_col == field_row:
                        if m == 0:
                            bc = {'r':{0:10}, 'z':{0:0}, 'priority':'z'}
                        else:
                            bc = {'r':{0:11}, 'z':{0:0}, 'priority':'r'}
                    elif field_row == ("velocity","pol") and field_col == field_row:
                        if m == 0:
                            bc = {'r':{0:20}, 'z':{0:0}, 'priority':'z'}
                        else:
                            bc = {'r':{0:10}, 'z':{0:0}, 'priority':'r'}
                    elif field_row == ("magnetic","tor") and field_col == field_row:
                        if m == 0:
                            bc = {'r':{0:10}, 'z':{0:20}, 'priority':'z'}
                        else:
                            bc = {'r':{0:0}, 'z':{0:20}, 'priority':'z'}
                    elif field_row == ("magnetic","pol") and field_col == field_row:
                        if m == 0:
                            bc = {'r':{0:20}, 'z':{0:40}, 'priority':'z'}
                        else:
                            bc = {'r':{0:10}, 'z':{0:40}, 'priority':'z'}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {'r':{0:10}, 'z':{0:20}, 'priority':'z'}

            # Stress-free/No-slip, Fixed flux/Fixed temperature
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("temperature",""):
                        bc = {'r':{0:-11, 'rt':0}, 'z':{0:-20, 'rt':0}}

                else:
                    if field_row == ("temperature","") and field_col == field_row:
                        bc = {'r':{0:11}, 'z':{0:20}, 'priority':'z'}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocity","tor"):
                    bc['r']['rt'] = 1
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","pol"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 4
                elif field_row == ("temperature",""):
                    bc['r']['rt'] = 1
                    bc['z']['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocity","tor"):
                        if m == 0:
                            bc = {'r':{0:-10, 'rt':1}, 'z':{0:-20, 'rt':2}}
                        else:
                            bc = {'r':{0:-11, 'rt':1}, 'z':{0:-20, 'rt':2}}
                    elif field_col == ("velocity","pol"):
                        if m == 0:
                            bc = {'r':{0:-20, 'rt':2}, 'z':{0:-40, 'rt':4}}
                        else:
                            bc = {'r':{0:-22, 'rt':2}, 'z':{0:-40, 'rt':4}}
                    elif field_col == ("temperature",""):
                        bc = {'r':{0:-10, 'rt':1}, 'z':{0:-20, 'rt':2}}

                elif bcId == 1:
                    if field_col == ("temperature",""):
                        bc = {'r':{0:-11, 'rt':1}, 'z':{0:-20, 'rt':2}}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocity","tor"):
                    bc['r']['rt'] = 1
                    bc['z']['rt'] = 2
                elif field_row == ("velocity","pol"):
                    bc['r']['rt'] = 2
                    bc['z']['rt'] = 4
                elif field_row == ("temperature",""):
                    bc['r']['rt'] = 1
                    bc['z']['rt'] = 2

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        Ra = eq_params['rayleigh']
        G = eq_params['gamma']
        zscale = eq_params['scale3d']
        m = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","pol") and field_col == ("temperature",""):
            if m == 0:
                mat = geo.i4drj4(res[0], res[2], 1, bc, Ra*G**3, restriction = restriction)
            else:
                mat = geo.i6laplhj4(res[0], res[2], m, bc, Ra*G**3, restriction = restriction)


        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        m = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("temperature","") and field_col == field_row:
            mat = geo.i2j2(res[0], res[2], m, bc, restriction = restriction)

        elif field_row == ("velocity","tor") and field_col == field_row:
            if m == 0:
                mat = geo.qid(res[0], res[2], 1, 1, 2, bc, -1.0, restriction = restriction)
            else:
                mat = geo.qid(res[0], res[2], m, 2, 2, bc, -1.0, restriction = restriction)
                #mat = geo.zblk(res[0], res[2], m, 2, 2, bc, restriction = restriction)

        elif field_row == ("velocity","pol") and field_col == field_row:
            if m == 0:
                mat = geo.qid(res[0], res[2], 1, 2, 4, bc, restriction = restriction)
            else:
                mat = geo.qid(res[0], res[2], m, 3, 4, bc, restriction = restriction)
                #mat = geo.zblk(res[0], res[2], m, 3, 4, bc, restriction = restriction)


        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        G = eq_params['gamma']
        zscale = eq_params['scale3d']
        m = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("magnetic","tor"):
            if field_col == ("velocity","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 2, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("velocity","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 2, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("magnetic","tor"):
                if m == 0:
                    mat = geo.i2j2lapl(res[0], res[2], 1, bc, zscale = zscale, restriction = restriction)
                else:
                    mat = geo.i2j2lapl(res[0], res[2], m, bc, zscale = zscale, restriction = restriction)

            elif field_col == ("magnetic","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 2, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("temperature",""):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 0, 1, 2, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

        elif field_row == ("magnetic","pol"):
            if field_col == ("velocity","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 2, 4, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 2, 4, bc, restriction = restriction)

            elif field_col == ("velocity","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 2, 4, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 2, 4, bc, restriction = restriction)

            elif field_col == ("magnetic","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 2, 4, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 2, 4, bc, restriction = restriction)

            elif field_col == ("magnetic","pol"):
                if m == 0:
                    mat = geo.i4j4lapl2(res[0], res[2], 1, bc, zscale = zscale, restriction = restriction)
                else:
                    mat = geo.i4j4lapl2(res[0], res[2], m, bc, zscale = zscale, restriction = restriction)

            elif field_col == ("temperature",""):
                if self.linearize:
                    if m == 0:
                        mat = geo.i4drj4(res[0], res[2], 1, bc, -Ra*G**3, restriction = restriction)
                    else:
                        mat = geo.i4laplhj4(res[0], res[2], m, bc, -Ra*G**3, restriction = restriction)
                else:
                    if m == 0:
                        mat = geo.zblk(res[0], res[2], 1, 2, 4, bc, restriction = restriction)
                    else:
                        mat = geo.zblk(res[0], res[2], m, 2, 4, bc, restriction = restriction)

        elif field_row == ("velocity","tor"):
            if field_col == ("velocity","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.i2laplh(res[0], res[2], m, bc, restriction = restriction)

            elif field_col == ("velocity","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

            elif field_col == ("magnetic","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.i2(res[0], res[2], m, bc, -1.0, restriction = restriction)

            elif field_col == ("magnetic","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

            elif field_col == ("temperature",""):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

        elif field_row == ("velocity","pol"):
            if field_col == ("velocity","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

            elif field_col == ("velocity","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.i2laplh(res[0], res[2], m, bc, restriction = restriction)

            elif field_col == ("magnetic","tor"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

            elif field_col == ("magnetic","pol"):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.i2(res[0], res[2], m, bc, -1.0, restriction = restriction)

            elif field_col == ("temperature",""):
                if m == 0:
                    mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

        elif field_row == ("temperature",""):
            if field_col == ("velocity","tor"):
                mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("velocity","pol"):
                mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("magnetic","tor"):
                mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("magnetic","pol"):
                if self.linearize or bcs["bcType"] == self.FIELD_TO_RHS:
                    if m == 0:
                        mat = geo.i2r_1drj2(res[0], res[2], m, bc, -G, restriction = restriction)
                    else:
                        mat = geo.i2j2(res[0], res[2], m, bc, -G, restriction = restriction)
                else:
                    mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction)

            elif field_col == ("temperature",""):
                mat = geo.i2j2lapl(res[0], res[2], m, bc, 1.0, zscale = zscale, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        Pr = eq_params['prandtl']
        zscale = eq_params['scale3d']
        m = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("magnetic","tor"):
            if m == 0:
                mat = geo.i2j2(res[0], res[2], 1, bc, 1.0/Pr, restriction = restriction)
            else:
                mat = geo.i2j2(res[0], res[2], m, bc, 1.0/Pr, restriction = restriction)

        elif field_row == ("magnetic","pol"):
            if m == 0:
                mat = geo.i4j4lapl(res[0], res[2], 1, bc, 1.0/Pr, zscale = zscale, restriction = restriction)
            else:
                mat = geo.i4j4lapl(res[0], res[2], m, bc, 1.0/Pr, zscale = zscale, restriction = restriction)

        elif field_row == ("velocity","tor"):
            if m == 0:
                mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
            else:
                mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

        elif field_row == ("velocity","pol"):
            if m == 0:
                mat = geo.zblk(res[0], res[2], 1, 1, 0, bc, restriction = restriction)
            else:
                mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction)

        elif field_row == ("temperature",""):
            mat = geo.i2j2(res[0], res[2], m, bc, restriction = restriction)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block of boundary operator"""

        assert(eigs[0].is_integer())

        m = int(eigs[0])

        zscale = eq_params['scale3d']
        Ra = eq_params['rayleigh']
        G = eq_params['gamma']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if self.use_galerkin:
            bc['priority'] = 'z'
        else:
            bc['priority'] = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)['priority']
        if field_row == ("velocity","tor"):
            mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction) 
        elif field_row == ("velocity","pol"):
            mat = geo.zblk(res[0], res[2], m, 1, 0, bc, restriction = restriction) 
        elif field_row == ("magnetic","tor"):
            mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction) 
            bcnr, bcnz, pad = (1, 2, 0)
            if bc['priority'] in ['r','sr']:
                kr = bcnz
            elif bc['priority'] in ['z','sz']:
                kr = 0
            mat = geo.zblk(res[0], res[2], m, bcnr, bcnz, bc, restriction = restriction) 
            if field_col == ("velocity","tor"):
                bc['r'][0] = min(0, bc['r'][0])
                bc['z'][0] = min(0, bc['z'][0])
                mat += geo.tau_mat_r(res[0], res[2], m, {0:10, 'c':1j*m, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i2), bcnr, bcnz, bc, restriction = restriction)

            elif field_col == ("velocity","pol"):
                bc['r'][0] = min(0, bc['r'][0])
                bc['z'][0] = min(0, bc['z'][0])
                mat += geo.tau_mat_r(res[0], res[2], m, {0:11, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i2d1, cscale = zscale), bcnr, bcnz, bc, restriction = restriction)
        elif field_row == ("magnetic","pol"):
            mat = geo.zblk(res[0], res[2], m, 2, 4, bc, restriction = restriction) 
            bcnr, bcnz, pad = (2, 4, 1)
            if bc['priority'] in ['r', 'sr']:
                kr = bcnz
            elif bc['priority'] in ['z', 'sz']:
                kr = 0
            mat = geo.zblk(res[0], res[2], m, bcnr, bcnz, bc, restriction = restriction) 
            if field_col == ("magnetic","tor"):
                bc['r'][0] = min(0, bc['r'][0])
                bc['z'][0] = min(0, bc['z'][0])
                mat += geo.tau_mat_r(res[0], res[2], m, {0:11, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4d1, cscale = zscale), bcnr, bcnz, bc, restriction = restriction)

            elif field_col == ("magnetic","pol"):
                bc['r'][0] = min(0, bc['r'][0])
                bc['z'][0] = min(0, bc['z'][0])
                mat += geo.tau_mat_r(res[0], res[2], m, {0:15, 'c':-1j*m, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4), bcnr, bcnz, bc, restriction = restriction)
                mat += geo.tau_mat_r(res[0], res[2], m, {0:10, 'c':-1j*m, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4d2, cscale = zscale), bcnr, bcnz, bc, restriction = restriction)

            elif field_col == ("temperature",""):
                bc['r'][0] = min(0, bc['r'][0])
                bc['z'][0] = min(0, bc['z'][0])
                mat += geo.tau_mat_r(res[0], res[2], m, {0:10, 'c':1j*m*Ra*G**3, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4), bcnr, bcnz, bc, restriction = restriction)
        elif field_row == ("temperature",""):
            mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction) 

#        if m == 0:
#            if field_col == ("velocity","tor"):
#                mat = geo.zblk(res[0], res[2], 1, 1, 2, bc, restriction = restriction) 
#            elif field_col == ("velocity","pol"):
#                mat = geo.zblk(res[0], res[2], 1, 2, 4, bc, restriction = restriction) 
#            elif field_col == ("temperature",""):
#                mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction) 
#        else:
#            if self.use_galerkin:
#                bc['priority'] = 'z'
#            else:
#                bc['priority'] = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)['priority']
#            if field_row == ("velocity","pol"):
#                bcnr, bcnz, pad = (3, 4, 2)
#                if bc['priority'] in ['r', 'sr']:
#                    kr = bcnz
#                    kz = 0
#                elif bc['priority'] in ['z', 'sz']:
#                    kr = 0
#                    kz = bcnr
#                mat = geo.zblk(res[0], res[2], m, bcnr, bcnz, bc, restriction = restriction) 
#                if field_col == ("velocity","tor"):
#                    bc['r'][0] = min(0, bc['r'][0])
#                    bc['z'][0] = min(0, bc['z'][0])
#                    mat += geo.tau_mat_r(res[0], res[2], m, {0:16, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4d1, cscale = zscale), bcnr, bcnz, bc, restriction = restriction)
#
#                elif field_col == ("velocity","pol"):
#                    bc['r'][0] = min(0, bc['r'][0])
#                    needZ = (bc['z'][0] >= 0 and bc['z'][0] < 40)
#                    bc['z'][0] = min(0, bc['z'][0])
#                    mat += geo.tau_mat_r(res[0], res[2], m, {0:17, 'c':-1j*m, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4), bcnr, bcnz, bc, restriction = restriction)
#                    mat += geo.tau_mat_r(res[0], res[2], m, {0:15, 'c':-1j*m, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4d2, cscale = zscale), bcnr, bcnz, bc, restriction = restriction)
#
#                    if needZ:
#                        mat += geo.tau_mat_z(res[0], res[2], m, {0:40, 'kron_shift':kz}, functools.partial(geo.rad.i6laplh), bcnr, bcnz, bc, restriction = restriction)
#
#                elif field_col == ("temperature",""):
#                    bc['r'][0] = min(0, bc['r'][0])
#                    bc['z'][0] = min(0, bc['z'][0])
#                    mat += geo.tau_mat_r(res[0], res[2], m, {0:10, 'c':1j*m*Ra*G**3, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i4), bcnr, bcnz, bc, restriction = restriction)
#
#            elif field_row == ("velocity","tor"):
#                bcnr, bcnz, pad = (2, 2, 1)
#                if bc['priority'] in ['r','sr']:
#                    kr = bcnz
#                    kz = 0
#                elif bc['priority'] in ['z','sz']:
#                    kr = 0
#                    kz = bcnr
#                mat = geo.zblk(res[0], res[2], m, bcnr, bcnz, bc, restriction = restriction) 
#                if field_col == ("velocity","tor"):
#                    bc['r'][0] = min(0, bc['r'][0])
#                    needZ = (bc['z'][0] >= 0 and bc['z'][0] < 20)
#                    bc['z'][0] = min(0, bc['z'][0])
#                    mat += geo.tau_mat_r(res[0], res[2], m, {0:10, 'c':1j*m, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i2), bcnr, bcnz, bc, restriction = restriction)
#
#                    if needZ:
#                        mat += geo.tau_mat_z(res[0], res[2], m, {0:20, 'kron_shift':kz}, functools.partial(geo.rad.i4laplh), bcnr, bcnz, bc, restriction = restriction)
#
#                elif field_col == ("velocity","pol"):
#                    bc['r'][0] = min(0, bc['r'][0])
#                    bc['z'][0] = min(0, bc['z'][0])
#                    mat += geo.tau_mat_r(res[0], res[2], m, {0:11, 'pad':pad, 'kron_shift':kr}, functools.partial(geo.c1d.i2d1, cscale = zscale), bcnr, bcnz, bc, restriction = restriction)
#
#            elif field_row == ("temperature",""):
#                mat = geo.zblk(res[0], res[2], m, 1, 2, bc, restriction = restriction) 

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat


class BoussinesqRBCCylinderVisu(BoussinesqRBCCylinderConfig, base_model.BaseModel):
    """Class to setup the Boussinesq Rayleigh-Benard convection in a cylinder (Toroidal/Poloidal formulation)"""

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        fields =  []

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            if field_row in [("mean_temperature", ""),("fluct_temperature", "")]:
                fields = [("temperature","")]
            elif field_row in [("magnetic", "tor")]:
                fields = [("velocity", "tor")]
            elif field_row in [("magnetic", "pol")]:
                fields = [("velocity", "pol")]
            else:
                fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]*res[2]
        if self.use_galerkin:
            if field_row == ("velocity","tor"):
                shift_r = 1
                shift_z = 2
            elif field_row == ("velocity","pol"):
                shift_r = 2
                shift_z = 4
            elif field_row == ("temperature",""):
                shift_r = 1
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

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            raise RuntimeError("Equations are not setup properly!")

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            raise RuntimeError("Equations are not setup properly!")

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                raise RuntimeError("Equations are not setup properly!")

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        zscale = eq_params['scale3d']
        m = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("mean_temperature","") and field_col == ("temperature",""):
            if m == 0:
                mat = geo.qid(res[0], res[2], m, 0, 0, bc)
            else:
                mat = geo.zblk(res[0], res[2], m, 1, 2, bc)

        elif field_row == ("fluct_temperature","") and field_col == ("temperature",""):
            if m == 0:
                mat = geo.zblk(res[0], res[2], m, 0, 0, bc)
            else:
                mat = geo.qid(res[0], res[2], m, 1, 2, bc)

        elif field_row == ("magnetic","tor") and field_col == ("velocity","tor"):
            if m == 0:
                mat = geo.qid(res[0], res[2], m, 0, 0, bc)
            else:
                mat = geo.zblk(res[0], res[2], m, 2, 2, bc)

        elif field_row == ("magnetic","pol") and field_col == ("velocity","pol"):
            if m == 0:
                mat = geo.qid(res[0], res[2], m, 0, 0, bc)
            else:
                mat = geo.zblk(res[0], res[2], m, 3, 4, bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
