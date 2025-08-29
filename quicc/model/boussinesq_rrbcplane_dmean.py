"""Module provides the functions to generate the Boussinesq rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (Toroidal/poloidal formulation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class BoussinesqRRBCPlane(base_model.BaseModel):
    """Class to setup the Boussinesq rotating Rayleigh-Benard convection in a plane layer (2 periodic directions) (Toroidal/poloidal formulation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "ekman", "scale1d"]

    def automatic_parameters(self, eq_params):
        """Extend parameters with automatically computable values"""

        # Rescale Z direction with ekman number
        d = {"scale1d":eq_params["scale1d"]*eq_params["ekman"]**(1./3.)}

        return d

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocity", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("velocity","tor"), ("velocity","pol"), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row in [("velocity","tor"), ("velocity","pol"), ("temperature","")]:
            fields =  [("velocity","tor"), ("velocity","pol"), ("temperature","")]
        else:
            fields = [field_row]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
            if field_row in [("temperature",""),("dz_meantemperature","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            fields = []

        return fields

    def block_size(self, res, eigs, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row in [("velocity","tor"), ("temperature","")]:
                shift_z = 2
            elif field_row in [("velocity","pol")]:
                shift_z = 4
            else:
                shift_z = 0

            gal_n = (res[0] - shift_z)

        else:
            gal_n = tau_n
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_z,0,0), 1)
        return block_info

    def stencil(self, res, eq_params, eigs, bcs, field_row, make_square):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return geo.stencil(res[0], bc, make_square)

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        is_complex = False

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.MODE

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

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
            # No-slip / Fixed temperature
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("velocity","tor"):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocity","pol"):
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:-20, 'rt':0}
                        else:
                            bc = {0:-40, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}

                else:
                    if field_row == ("velocity","tor") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("velocity","pol") and field_col == field_row:
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:20}
                        else:
                            bc = {0:40}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {0:20}

            # Stress-free / Fixed flux
            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("velocity","tor"):
                        bc = {0:-21, 'rt':0}
                    elif field_col == ("velocity","pol"):
                        bc = {0:-41, 'rt':0}
                    elif field_col == ("temperature",""):
                        bc = {0:-21, 'rt':0}

                else:
                    if field_row == ("velocity","tor") and field_col == field_row:
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:21}
                        else:
                            bc = {0:21}
                    elif field_row == ("velocity","pol") and field_col == field_row:
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:21}
                        else:
                            bc = {0:41}
                    elif field_row == ("temperature","") and field_col == field_row:
                        bc = {0:21}

            # Ekman-pumping
            elif bcId == 2:
                if self.use_galerkin:
                    raise RuntimeError("Equations are not setup properly!")

                else:
                    if field_row == ("velocity","tor") and field_col == field_row:
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:21}
                        else:
                            bc = {0:21}
                    elif field_row == ("velocity","pol") and field_col == field_row:
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:21}
                        else:
                            bc = {0:41, 'use_parity':True}
                    elif field_row == ("velocity","pol") and field_col == ("velocity","tor"):
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:0}
                        else:
                            E = eq_params['ekman']
                            c = E**(1./6.)/2.0**(1./2.)
                            bc = {0:20, 'c':[c, -c], 'use_parity':True}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocity","tor"):
                    bc['rt'] = 2
                elif field_row == ("velocity","pol"):
                    if eigs[0] == 0 and eigs[1] == 0:
                        bc['rt'] = 2
                    else:
                        bc['rt'] = 4
                elif field_row in [("temperature","")]:
                    bc['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if field_col == ("velocity","tor"):
                        bc = {0:-20}
                    elif field_col == ("velocity","pol"):
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:-20}
                        else:
                            bc = {0:-40}
                    elif field_col in [("temperature","")]:
                        bc = {0:-20}

                elif bcId == 1:
                    if field_col == ("velocity","tor"):
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:-21}
                        else:
                            bc = {0:-21}
                    elif field_col == ("velocity","pol"):
                        if eigs[0] == 0 and eigs[1] == 0:
                            bc = {0:-21}
                        else:
                            bc = {0:-41}
                    elif field_col in [("temperature","")]:
                        bc = {0:-21}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                if field_row == ("velocity","tor"):
                    bc['rt'] = 2
                elif field_row == ("velocity","pol"):
                    if eigs[0] == 0 and eigs[1] == 0:
                        bc['rt'] = 2
                    else:
                        bc['rt'] = 4
                elif field_row in [("temperature","")]:
                    bc['rt'] = 2

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        kx = eigs[0]
        ky = eigs[1]
        laplh = -(kx**2 + ky**2)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create the explicit nonlinear operator"""

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("temperature","") and field_col == field_row:
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0], bc)
            else:
                mat = geo.i2(res[0], bc)

        elif field_row == ("dz_meantemperature","") and field_col == field_row:
            if kx == 0 and ky == 0:
                mat = (geo.qid(res[0],0, bc) - spsp.eye(res[0], 1)*geo.avg(res[0]))
            else:
                mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Ra = eq_params['rayleigh']
        Pr = eq_params['prandtl']
        E = eq_params['ekman']
        Ro = E**(1./3.)
        zscale = eq_params['scale1d']

        kx = eigs[0]
        ky = eigs[1]
        laplh = -(kx**2 + ky**2)

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        # Mean X velocity
        if field_row == ("velocity","tor") and kx == 0 and ky == 0:
            if field_col == ("velocity","tor"):
                mat = geo.i2d2(res[0], bc, 1.0, cscale = zscale)

            elif field_col == ("velocity","pol"):
                mat = geo.i2(res[0], bc, -1.0/Ro)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

        # Mean Y velocity
        elif field_row == ("velocity","pol") and kx == 0 and ky == 0:
            if field_col == ("velocity","tor"):
                mat = geo.i2(res[0], bc, 1.0/Ro)

            elif field_col == ("velocity","pol"):
                mat = geo.i2d2(res[0], bc, 1.0, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)
    
        # Temperature kx = 0, ky = 0
        elif field_row == ("temperature","") and kx == 0 and ky == 0:
            if field_col == ("velocity","tor"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","pol"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

        # Toroidal velocity
        elif field_row == ("velocity","tor"):
            if field_col == ("velocity","tor"):
                mat = geo.i2lapl(res[0], kx, ky, bc, laplh, cscale = zscale)

            elif field_col == ("velocity","pol"):
                mat = geo.i2d1(res[0], bc, laplh/Ro, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)

        # Poloidal velocity
        elif field_row == ("velocity","pol"):
            if field_col == ("velocity","tor"):
                mat = geo.i4d1(res[0], bc, -laplh/Ro, cscale = zscale)

            elif field_col == ("velocity","pol"):
                mat = geo.i4lapl2(res[0], kx, ky, bc, laplh, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.i4(res[0], bc, -laplh*(Ra/Pr))

        # temperature
        elif field_row == ("temperature",""):
            if field_col == ("velocity","tor"):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocity","pol"):
                if self.linearize or bcs["bcType"] == self.FIELD_TO_RHS:
                    mat = geo.i2(res[0], bc, -laplh)
                else:
                    mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = geo.i2lapl(res[0], kx, ky, bc, 1.0/Pr, cscale = zscale)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        kx = eigs[0]
        ky = eigs[1]
        laplh = -(kx**2 + ky**2)
        zscale = eq_params['scale1d']

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        # Mean X velocity
        if field_row == ("velocity","tor") and kx == 0 and ky == 0:
            mat = geo.i2(res[0], bc, 1.0)

        # Mean Y velocity
        elif field_row == ("velocity","pol") and kx == 0 and ky == 0:
            mat = geo.i2(res[0], bc, 1.0)

        # Temperature kx = 0, ky = 0 (fake timestepping, just keep it a zero)
        elif field_row == ("temperature","") and kx == 0 and ky == 0:
            mat = geo.qid(res[0], 2, bc)

        # Toroidal velocity
        elif field_row == ("velocity","tor"):
            mat = geo.i2(res[0], bc, laplh)

        # Poloidal velocity
        elif field_row == ("velocity","pol"):
            mat = geo.i4lapl(res[0], kx, ky, bc, laplh, cscale = zscale)

        # Temperature
        elif field_row == ("temperature",""):
            mat = geo.i2(res[0], bc, 1.0)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block of boundary operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        mat = geo.zblk(res[0], bc)
        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
