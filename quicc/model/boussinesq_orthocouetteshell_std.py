"""Module provides the functions to generate the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius as geo
import quicc.base.base_model as base_model
from quicc.geometry.spherical.shell_radius_boundary import no_bc


class BoussinesqOrthoCouetteShellStdConfig:
    """Class to setup the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, False, False]

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["ekman", "rossby", "rratio"]

    def automatic_parameters(self, eq_params):
        """Extend parameters with automatically computable values"""

        d = {"ro":1.0/(1.0 - eq_params["rratio"])}

        return d

    def config_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["velocity"]

    def stencil(self, res, eq_params, eigs, bcs, field_row, make_square):
        """Create the galerkin stencil"""
        
        assert(eigs[0].is_integer())
        l = eigs[0]

        # Get boundary condition
        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        mat = geo.stencil(res[0], bc, make_square)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is real
        is_complex = False

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.SLOWEST_MULTI_RHS

        return self.compile_equation_info(res, field_row, is_complex, index_mode)


class BoussinesqOrthoCouetteShellStd(BoussinesqOrthoCouetteShellStdConfig, base_model.BaseModel):
    """Class to setup the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""
    
        fields = [field_row]

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit linear dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
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

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("velocity","tor"):
                shift_r = 2
            elif field_row == ("velocity","pol"):
                shift_r = 4
            else:
                shift_r = 0

            gal_n = (res[0] - shift_r)

        else:
            gal_n = tau_n
            shift_r = 0

        block_info = (tau_n, gal_n, (shift_r,0,0), 1)
        return block_info

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        sgn = np.sign(eq_params['rossby'])
        # modified by Nicolò Lardelli -> use Ro=0 to compute stationary solution U_0
        sgn = 1 if sgn==0 else sgn
        ro = self.automatic_parameters(eq_params)['ro']
        ri = ro*eq_params['rratio']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])
        assert(eigs[0].is_integer())
        l = eigs[0]

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            bc = no_bc()
            bcId = bcs.get(field_col[0], -1)
            
            if bcId == 0:
                if self.use_galerkin:
                    raise RuntimeError("Inhomogeneous boundary conditions cannot use Galerkin scheme!")

                else:
                    if field_row == ("velocity","tor") and field_col == field_row:
                        bc = {0:20}
                    elif field_row == ("velocity","pol") and field_col == field_row:
                        bc = {0:40, 'c':{'a':a, 'b':b}}
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot used Galerkin scheme!")

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot used Galerkin scheme!")
        
        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot used Galerkin scheme!")

        else:
            bc = no_bc()
            

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        assert(eigs[0].is_integer())
        l = eigs[0]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def nonlinear_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit nonlinear term"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def implicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        E = eq_params['ekman']
        assert(eigs[0].is_integer())
        l = eigs[0]

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("velocity","tor") and field_col == field_row:
            mat = geo.i2r2lapl(res[0], l, a, b, bc, l*(l+1.0)*E)

        elif field_row == ("velocity","pol") and field_col == field_row:
            mat = geo.i4r4lapl2(res[0], l, a, b, bc, l*(l+1.0)*E)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row, restriction = None):
        """Create matrix block of time operator"""

        assert(eigs[0].is_integer())
        l = eigs[0]

        ro = self.automatic_parameters(eq_params)['ro']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("velocity","tor"):
            mat = geo.i2r2(res[0], a, b, bc, l*(l+1.0))

        elif field_row == ("velocity","pol"):
            mat = geo.i4r4lapl(res[0], l, a, b, bc, l*(l+1.0))

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat

    def boundary_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block linear operator"""

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat


class BoussinesqOrthoCouetteShellStdVisu(BoussinesqOrthoCouetteShellStdConfig, base_model.BaseModel):
    """Class to setup the Boussinesq spherical Couette in a spherical shell (Toroidal/Poloidal formulation) without field coupling (standard implementation)"""

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""
    
        fields = []

        return fields

    def explicit_fields(self, timing, field_row):
        """Get the list of fields with explicit linear dependence"""

        # Explicit linear terms
        if timing == self.EXPLICIT_LINEAR:
            if field_row in [("zonal_velocity","tor"),("nonzonal_velocity","tor")]:
                fields = [("velocity","tor")]
            elif field_row in [("zonal_velocity","pol"),("nonzonal_velocity","pol")]:
                fields = [("velocity","pol")]
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

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("velocity","tor"):
                shift_r = 2
            elif field_row == ("velocity","pol"):
                shift_r = 4
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

        # Matrix operator is real
        is_complex = False

        # Index mode: SLOWEST_SINGLE_RHS, SLOWEST_MULTI_RHS, MODE, SINGLE
        index_mode = self.MODE

        return self.compile_equation_info(res, field_row, is_complex, index_mode)

    def convert_bc(self, eq_params, eigs, bcs, field_row, field_col):
        """Convert simulation input boundary conditions to ID"""

        sgn = np.sign(eq_params['rossby'])
        ro = self.automatic_parameters(eq_params)['ro']
        ri = ro*eq_params['rratio']
        a, b = geo.linear_r2x(ro, eq_params['rratio'])

        assert(eigs[0].is_integer())
        l = eigs[0]

        # Solver: no tau boundary conditions
        if bcs["bcType"] == self.SOLVER_NO_TAU and not self.use_galerkin:
            bc = no_bc()

        # Solver: tau and Galerkin
        elif bcs["bcType"] == self.SOLVER_HAS_BC or bcs["bcType"] == self.SOLVER_NO_TAU:
            bc = no_bc()
            
            # Set LHS galerkin restriction
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot used Galerkin scheme!")

        # Stencil:
        elif bcs["bcType"] == self.STENCIL:
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot used Galerkin scheme!")
        
        # Field values to RHS:
        elif bcs["bcType"] == self.FIELD_TO_RHS:
            bc = no_bc()
            if self.use_galerkin:
                raise RuntimeError("Inhomogeneous boundary conditions cannot used Galerkin scheme!")

        else:
            bc = no_bc()

        return bc

    def explicit_block(self, res, eq_params, eigs, bcs, field_row, field_col, restriction = None):
        """Create matrix block for explicit linear term"""

        assert(eigs[0].is_integer())
        l = eigs[0]
        m = eigs[1]

        mat = None
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("zonal_velocity","tor") and field_col == ("velocity","tor"):
            if m == 0:
                mat = geo.qid(res[0], 0, bc)
            else:
                mat = geo.zblk(res[0], bc)
        elif field_row == ("zonal_velocity","pol") and field_col == ("velocity","pol"):
            if m == 0:
                mat = geo.qid(res[0], 0, bc)
            else:
                mat = geo.zblk(res[0], bc)
        elif field_row == ("nonzonal_velocity","tor") and field_col == ("velocity","tor"):
            if m > 0:
                mat = geo.qid(res[0], 0, bc)
            else:
                mat = geo.zblk(res[0], bc)
        elif field_row == ("nonzonal_velocity","pol") and field_col == ("velocity","pol"):
            if m > 0:
                mat = geo.qid(res[0], 0, bc)
            else:
                mat = geo.zblk(res[0], bc)

        if mat is None:
            raise RuntimeError("Equations are not setup properly!")

        return mat
