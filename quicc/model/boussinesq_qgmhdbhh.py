"""Module provides the functions to generate the Boussinesq F-Plane QG model with horizontal imposed magnetic field and low Rm"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as geo
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class BoussinesqQGmhdBhh(base_model.BaseModel):
    """Class to setup the Boussinesq F-Plane QG MHD Bhh model"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "magnetic_prandtl", "chandrasekhar", "rayleigh", "scale1d"]

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature"]
#        return ["streamfunction", "velocityz", "temperature", "bx", "by"] # Do I need this?

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("streamfunction",""), ("velocityz",""), ("temperature",""), ("fbx",""), ("fby",""), ("fbz","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row in [("streamfunction",""), ("velocityz",""), ("temperature","")]:
            fields =  [("streamfunction",""), ("velocityz",""), ("temperature","")]

        elif field_row in [("fbx","")]:
            fields =  [("fbx","")]

        elif field_row in [("fby","")]:
            fields =  [("fby","")]

        elif field_row in [("fbz","")]:
            fields =  [("fbz","")]

        elif field_row in [("bx","")]:
            fields =  [("bx","")]

        elif field_row in [("by","")]:
            fields =  [("by","")]

        else:
            fields = [field_row]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit linear dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            fields = []
#            if field_row in [("bx","")]: 
#                fields = [("emfy","")]
#            elif field_row in [("by","")]: 
#                fields = [("emfx","")]
#            else:
#                fields = []

        # Explicit nonlinear terms
        elif timing == self.EXPLICIT_NONLINEAR:
#            if field_row in [("temperature",""), ("streamfunction",""), ("velocityz",""), ("dz_meantemperature",""), ("fbx",""), ("fby",""), ("fbz",""), ("emfx",""), ("emfy",""), ("pressure","")]:
            if field_row in [("temperature",""), ("streamfunction",""), ("velocityz",""), ("dz_meantemperature",""), ("fbx",""), ("fby",""), ("fbz",""), ("dissTh",""), ("dissV",""), ("dissB","")]:
                fields = [field_row]
            else:
                fields = []

        # Explicit update terms for next step
        elif timing == self.EXPLICIT_NEXTSTEP:
            if field_row == ("vorticityz",""):
                fields = [("streamfunction","")]
            elif field_row == ("velocityx",""):
                fields = [("streamfunction","")]
            elif field_row == ("velocityy",""):
                fields = [("streamfunction","")]
            elif field_row == ("fjz",""):
                fields = [("fbx",""), ("fby","")] # this line works fine
            else:
                fields = []

        return fields

    def block_size(self, res, eig, bcs, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("velocityz",""):
                shift_z = 2
            elif field_row == ("streamfunction",""):
                shift_z = 2
            elif field_row == ("temperature",""):
                shift_z = 2
            else:
                shift_z = 0

            gal_n = res[0] - shift_z 

        else:
            gal_n = tau_n
            shift_z = 0

        block_info = (tau_n, gal_n, (shift_z,0,0), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
        is_complex = True

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
            kx = eigs[0]
            ky = eigs[1]

            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            if bcId == 0:
                if self.use_galerkin:
                    if field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}
                    elif field_col == ("velocityz",""):
                        bc = {0:-20, 'rt':0}

                else:
                    if bcs["bcType"] == self.SOLVER_HAS_BC:
                        if field_row == ("velocityz","") and field_col == field_row:
                            bc = {0:11}
                        elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                            bc = {0:10}
                        elif field_row == ("temperature","") and field_col == field_row:
                            bc = {0:991}
#                        elif field_row == ("bx","") and field_col == ("bx",""):
#                            bc = {0:20}
#                        elif field_row == ("by","") and field_col == ("by",""):
#                            bc = {0:20}
                    else:
                        bc = no_bc()

            elif bcId == 1:
                if self.use_galerkin:
                    if field_col == ("temperature",""):
                        bc = {0:-20, 'rt':0}

                else:
                    if bcs["bcType"] == self.SOLVER_HAS_BC:
                        if field_row == ("temperature","") and field_col == field_row:
                            bc = {0:20}

            # Set LHS galerkin restriction
            if self.use_galerkin:
                if field_row == ("velocityz","") or field_row == ("streamfunction",""):
                    bc['rt'] = 1
                elif field_row == ("temperature",""):
                    bc['rt'] = 2

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                if field_col == ("temperature",""):
                    bc = {0:-20, 'rt':0}
            elif field_col == ("velocityz",""):
                bc = {0:-20, 'rt':0}

        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                   if field_row == ("velocityz",""):
                       bc['rt'] = 2
                   elif field_row == ("temperature",""):
                       bc['rt'] = 2

        else:
            bc = no_bc()

        return bc

#    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
#        """Create matrix block for explicit linear term"""

#        tau = eq_params['tau']
#        zscale = eq_params['scale1d']

#        mat = None
#        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

#        if field_row == ("bx","") and field_col == ("emfy",""):
#            mat = geo.i2d1(res[0], bc, 1.0*tau, cscale = zscale)

#        elif field_row == ("by","") and field_col == ("emfx",""):
#            mat = geo.i2d1(res[0], bc, -1.0*tau, cscale = zscale)

#        if mat is None:
#            raise RuntimeError("Equations are not setup properly!")

#        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit nonlinear term"""

        zscale = eq_params['scale1d']

        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if field_row == ("temperature","") and field_col == field_row:
            if bcs["temperature"] == 1 and not self.use_galerkin:
                mat = geo.sid(res[0], 2, bc)
            else:
                mat = geo.sid(res[0], 1, bc)

        elif field_row == ("streamfunction","") and field_col == field_row:
            mat = geo.i1(res[0], bc)

        elif field_row == ("velocityz","") and field_col == field_row:
            mat = geo.i1(res[0], bc)

        elif field_row == ("dz_meantemperature","") and field_col == field_row:
            if eigs[0] == 0 and eigs[1] == 0:
                mat = (spsp.eye(res[0]) - spsp.eye(res[0], 1)*geo.avg(res[0]))
            else:
                mat = geo.zblk(res[0], bc)

#       For the dynamod model
#        elif field_row == ("emfx","") and field_col == field_row:
#            if eigs[0] == 0 and eigs[1] == 0:
#                mat = geo.qid(res[0], 0, bc)
#            else:
#                mat = geo.zblk(res[0], bc)

#        elif field_row == ("emfy","") and field_col == field_row:
#            if eigs[0] == 0 and eigs[1] == 0:
#                mat = geo.qid(res[0], 0, bc)
#            else:
#                mat = geo.zblk(res[0], bc)

#        elif field_row == ("pressure","") and field_col == field_row:
#            if eigs[0] == 0 and eigs[1] == 0:
#                mat = geo.qid(res[0], 0, bc)
#            else:
#                mat = geo.zblk(res[0], bc)


#       For the low Pm model
#        elif field_row == ("fbx","") and field_col == field_row:
#            if eigs[0] == 0 and eigs[1] == 0:
#                mat = geo.zblk(res[0],bc)
#            else:
#                mat = geo.qid(res[0], 0, bc, -1/(kx**2 + ky**2))

#        elif field_row == ("fby","") and field_col == field_row:
#            if eigs[0] == 0 and eigs[1] == 0:
#                mat = geo.zblk(res[0],bc)
#            else:
#                mat = geo.qid(res[0], 0, bc, -1/(kx**2 + ky**2))

#        elif field_row == ("fbz","") and field_col == field_row:
#            if eigs[0] == 0 and eigs[1] == 0:
#                mat = geo.zblk(res[0],bc)
#            else:
#                mat = geo.qid(res[0], 0, bc, -1/(kx**2 + ky**2))

        elif field_row == ("fbx","") and field_col == field_row:
            if eigs[0] == 0 and eigs[1] == 0:
                mat = geo.zblk(res[0],bc)
            else:
                mat = geo.qid(res[0], 0, bc, 1)

        elif field_row == ("fby","") and field_col == field_row:
            if eigs[0] == 0 and eigs[1] == 0:
                mat = geo.zblk(res[0],bc)
            else:
                mat = geo.qid(res[0], 0, bc, 1)

        elif field_row == ("fbz","") and field_col == field_row:
            if eigs[0] == 0 and eigs[1] == 0:
                mat = geo.zblk(res[0],bc)
            else:
                mat = geo.qid(res[0], 0, bc, 1)

        elif field_row == ("dissTh","") and field_col == field_row:
                mat = geo.qid(res[0], 0, bc, 1)

        elif field_row == ("dissV","") and field_col == field_row:
                mat = geo.qid(res[0], 0, bc, 1)

        elif field_row == ("dissB","") and field_col == field_row:
                mat = geo.qid(res[0], 0, bc, 1)  

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nextstep_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit nextstep update"""

        zscale = eq_params['scale1d']
        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("vorticityz","") and field_col == ("streamfunction",""):
            mat = geo.qid(res[0],0, bc, -(kx**2 + ky**2))
#       subtract barotropic component
#        if field_row == ("vorticityz","") and field_col == ("streamfunction",""):
#            mat = geo.qid(res[0],0,bc,-(kx**2+ky**2))  - spsp.eye(res[0],1)*geo.avg(res[0])*geo.qid(res[0],0,bc,-(kx**2+ky**2)) 

        elif field_row == ("fjz","") and field_col == ("fbx",""):
            mat = geo.qid(res[0],0, bc, -ky*1j)

        elif field_row == ("fjz","") and field_col == ("fby",""):
            mat = geo.qid(res[0],0, bc, kx*1j) 

        elif field_row == ("velocityx","") and field_col == ("streamfunction",""):
            mat = geo.qid(res[0],0, bc, -ky*1j)

        elif field_row == ("velocityy","") and field_col == ("streamfunction",""):
            mat = geo.qid(res[0],0, bc, kx*1j)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        MPr = eq_params['magnetic_prandtl']
        Ra = eq_params['rayleigh']
        zscale = eq_params['scale1d']
        kx = eigs[0]
        ky = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = geo.i1(res[0], bc, (kx**2 + ky**2)**2)

            elif field_col == ("velocityz",""):
                mat = geo.i1d1(res[0], bc, 1.0, cscale = zscale)

            elif field_col == ("temperature",""):
                mat = geo.zblk(res[0], bc)


        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = geo.i1d1(res[0], bc, -1.0, cscale = zscale)

            elif field_col == ("velocityz",""):
                mat = geo.i1(res[0], bc, -(kx**2 + ky**2))

            elif field_col == ("temperature",""):
                if kx == 0 and ky == 0:
                    mat = geo.zblk(res[0], bc)
                else:
                    mat = geo.i1(res[0], bc, (Ra/Pr))*utils.qid_from_idx(utils.qidx(res[0], res[0]-1), res[0])

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                mat = geo.zblk(res[0], bc)

            elif field_col == ("velocityz",""):
                if self.linearize:
                    if bcs["temperature"] == 1 and not self.use_galerkin:
                        mat = geo.sid(res[0],2, bc)
                    else:
                        mat = geo.sid(res[0],1, bc)
                else:
                    mat = geo.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                if bcs["temperature"] == 1 and not self.use_galerkin:
                    mat = geo.sid(res[0],2, bc, -(1.0/Pr)*(kx**2 + ky**2))
                else:
                    mat = geo.sid(res[0],1, bc, -(1.0/Pr)*(kx**2 + ky**2))

        elif field_row == ("fbx","") and field_col == field_row:
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0],bc)
            else:
                mat = geo.qid(res[0], 0, bc, -(kx**2 + ky**2))

        elif field_row == ("fby","") and field_col == field_row:
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0],bc)
            else:
                mat = geo.qid(res[0], 0, bc, -(kx**2 + ky**2))

        elif field_row == ("fbz","") and field_col == field_row:
            if kx == 0 and ky == 0:
                mat = geo.zblk(res[0],bc)
            else:
                mat = geo.qid(res[0], 0, bc, -(kx**2 + ky**2))

#        elif field_row == ("bx",""):
#            if field_col == ("bx",""):
#                mat = geo.i2d2(res[0], bc, tau/MPr, cscale = zscale)

#        elif field_row == ("by",""):
#            if field_col == ("by",""):
#                mat = geo.i2d2(res[0], bc, tau/MPr, cscale = zscale)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        kx = eigs[0]
        ky = eigs[1]
        MPr = eq_params['magnetic_prandtl']


        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = geo.i1(res[0], bc, -(kx**2 + ky**2))

        elif field_row == ("velocityz",""):
            mat = geo.i1(res[0], bc)

        elif field_row == ("temperature",""):
            if bcs["temperature"] == 1 and not self.use_galerkin:
                mat = geo.sid(res[0],2, bc)
            else:
                mat = geo.sid(res[0],1, bc)

        elif field_row == ("fbx",""):
            mat = geo.qid(res[0], 0, bc, MPr)

        elif field_row == ("fby",""):
            mat = geo.qid(res[0], 0, bc, MPr)

        elif field_row == ("fbz",""):
            mat = geo.qid(res[0], 0, bc, MPr)

#        elif field_row == ("bx",""):
#            mat = geo.i2(res[0], bc)

#        elif field_row == ("by",""):
#            mat = geo.i2(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if field_row == ("temperature","") and field_col == field_row:
            mat = geo.zblk(res[0], bc, location = 'b')
        else:
            mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
