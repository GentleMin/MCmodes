"""Module provides the functions to generate the Boussinesq F-Plane 3DQG model"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.cartesian.cartesian_1d as c1d
import quicc.base.base_model as base_model
from quicc.geometry.cartesian.cartesian_boundary_1d import no_bc


class BoussinesqFPlane3DQG(base_model.BaseModel):
    """Class to setup the Boussinesq F-Plane 3DQG model"""

    force_temperature_bc = True

    def nondimensional_parameters(self):
        """Get the list of nondimensional parameters"""

        return ["prandtl", "rayleigh", "theta"]

    def periodicity(self):
        """Get the domain periodicity"""

        return [False, True, True]

    def all_fields(self):
        """Get the list of fields that need a configuration entry"""

        return ["streamfunction", "velocityz", "temperature"]

    def stability_fields(self):
        """Get the list of fields needed for linear stability calculations"""

        fields =  [("streamfunction",""), ("velocityz",""), ("temperature","")]

        return fields

    def implicit_fields(self, field_row):
        """Get the list of coupled fields in solve"""

        if field_row == ("vorticity",""):
            fields = [("vorticity","")]

        elif field_row == ("meantemperature",""):
            fields = [("meantemperature","")]

        else:
            fields =  [("streamfunction",""), ("velocityz",""), ("temperature","")]

        return fields

    def explicit_fields(self, field_row):
        """Get the list of fields with explicit linear dependence"""

        if field_row == ("vorticity",""):
            fields = [("streamfunction","")]

        else:
            fields = []

        return fields

    def block_size(self, res, field_row):
        """Create block size information"""

        tau_n = res[0]
        if self.use_galerkin:
            if field_row == ("streamfunction",""):
                shift_x = 0
            elif field_row == ("velocityz",""):
                shift_x = 0
            elif field_row == ("temperature",""):
                shift_x = 2
            else:
                shift_x = 0

            gal_n = res[0] - shift_x 

        else:
            gal_n = tau_n
            shift_x = 0

        block_info = (tau_n, gal_n, (shift_x,0,0), 1)
        return block_info

    def equation_info(self, res, field_row):
        """Provide description of the system of equation"""

        # Matrix operator is complex except for vorticity and mean temperature
        if field_row == ("vorticity","") or field_row == ("meantemperature",""):
            is_complex = False
        else:
            is_complex = True

        # Implicit field coupling
        im_fields = self.implicit_fields(field_row)
        # Additional explicit linear fields
        ex_fields = self.explicit_fields(field_row)

        # Index mode: SLOWEST = 0, MODE = 1, GEOMETRIC_1D_3D = 2
        index_mode = 1

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

        # Impose no boundary conditions
        if bcs["bcType"] == 2:
            bc = no_bc()

            if field_row ==("temperature",""):
                bc['r'] = 2
            elif eq_params['theta'] == 0:
                if field_row == ("streamfunction",""):
                    bc['r'] = 2
                elif field_row == ("velocityz",""):
                    bc['r'] = 2
        else:
            # Impose no tau boundary conditions
            if bcs["bcType"] == 1 and not self.use_galerkin:
                bc = no_bc()

            # Impose tau and Galerkin
            else:
                eta2 = np.sin(np.pi*eq_params['theta']/180)
                eta3 = np.cos(np.pi*eq_params['theta']/180)
                kx = eigs[0]
                ky = eigs[1]

                bc = no_bc()
                bcId = bcs.get(field_col[0], -1)
                if bcId == 0:
                    if self.use_galerkin:
                        if field_col == ("temperature",""):
                            bc = {0:-20, 'r':0}

                        elif eq_params['theta'] == 0:
                            if field_col == ("streamfunction",""):
                                bc = {0:-21, 'r':0}
                            elif field_col == ("velocityz",""):
                                bc = {0:-20, 'r':0}
                        else:
                            if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                                bc = {0:10, 'c':-1j*kx*eta2, 'r':0}
                            elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                                bc = {0:11, 'c':eta3, 'r':0}
                            elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                                bc = {0:10, 'c':eta3, 'r':0}
                            elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                                bc = {0:11, 'c':-1j*kx*eta2, 'r':0}

                        if field_row == ("temperature",""):
                            bc['r'] = 2
                        elif eq_params['theta'] == 0:
                            if field_row == ("streamfunction",""):
                                bc['r'] = 2
                            elif field_row == ("velocityz",""):
                                bc['r'] = 2

                    else:
                        if field_row == ("streamfunction","") and field_col == ("streamfunction",""):
                            bc = {0:10, 'c':-1j*kx*eta2}
                        elif field_row == ("velocityz","") and field_col == ("velocityz",""):
                            bc = {0:11, 'c':eta3}
                        elif field_row == ("streamfunction","") and field_col == ("velocityz",""):
                            bc = {0:10, 'c':eta3}
                        elif field_row == ("velocityz","") and field_col == ("streamfunction",""):
                            bc = {0:11, 'c':-1j*kx*eta2}

        return bc

    def stencil(self, res, eq_params, eigs, bcs, field_row):
        """Create the galerkin stencil"""
        
        # Get boundary condition
        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        return c1d.stencil(res[0], bc)

    def qi(self, res, eq_params, eigs, bcs, field_row):
        """Create the quasi-inverse operator"""

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = c1d.i1(res[0], bc)

        elif field_row == ("velocityz",""):
            mat = c1d.i1(res[0], bc)

        elif field_row == ("temperature",""):
            mat = c1d.qid(res[0], 0, bc)

            # Force temperature boundary condition
            if self.force_temperature_bc and not self.use_galerkin:
                mat = mat.tolil()
                mat[-2:,:] = 0
                mat = mat.tocsr()

        elif field_row == ("meantemperature",""):
            if eigs[0] == 0 and eigs[1] == 0:
                mat = (c1d.qid(res[0], 0,bc) - spsp.eye(res[0],1)*c1d.avg(res[0]))
            else:
                mat = c1d.zblk(res[0], bc)

        return mat

    def linear_block(self, res, eq_params, eigs, bcs, field_row, field_col):
        """Create matrix block linear operator"""

        Pr = eq_params['prandtl']
        Ra = eq_params['rayleigh']
        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        kx = eigs[0]
        ky = eigs[1]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_col)
        if field_row == ("streamfunction",""):
            if field_col == ("streamfunction",""):
                mat = c1d.i1(res[0], bc, (kx**2 + (1/eta3**2)*ky**2)**2)

            elif field_col == ("velocityz",""):
                mat = c1d.i1d1(res[0], bc, 2*eta3)

            elif field_col == ("temperature",""):
                mat = c1d.i1(res[0], bc, -1j*kx*eta2*(Ra/Pr))

        elif field_row == ("velocityz",""):
            if field_col == ("streamfunction",""):
                mat = c1d.i1d1(res[0], bc, -2*eta3)

            elif field_col == ("velocityz",""):
                mat = c1d.i1(res[0], bc, -(kx**2 + (1/eta3**2)*ky**2))

            elif field_col == ("temperature",""):
                mat = c1d.i1(res[0], bc, eta3*Ra/Pr)

        elif field_row == ("temperature",""):
            if field_col == ("streamfunction",""):
                if self.linearize:
                    mat = c1d.qid(res[0],0, bc, -1j*kx*eta2)

                    # Force temperature boundary condition
                    if self.force_temperature_bc and not self.use_galerkin:
                        mat = mat.tolil()
                        mat[-2:,:] = 0
                        mat = mat.tocsr()
                else:
                    mat = c1d.zblk(res[0], bc)

            elif field_col == ("velocityz",""):
                if self.linearize:
                    mat = c1d.qid(res[0],0, bc, eta3)

                    # Force temperature boundary condition
                    if self.force_temperature_bc and not self.use_galerkin:
                        mat = mat.tolil()
                        mat[-2:,:] = 0
                        mat = mat.tocsr()
                else:
                    mat = c1d.zblk(res[0], bc)

            elif field_col == ("temperature",""):
                mat = c1d.qid(res[0],0, bc, -(1/Pr)*(kx**2 + (1/eta3**2)*ky**2))

                # Force temperature boundary condition
                if self.force_temperature_bc and not self.use_galerkin:
                    mat = mat.tolil()
                    if bcs['bcType'] == 0:
                        tmp = c1d.qid(res[0],2,{0:20})
                    else:
                        tmp = c1d.qid(res[0],2, no_bc())
                    tmp = tmp.tolil()
                    mat[-2:,:] = tmp[0:2,:]
                    mat = mat.tocsr()

        elif field_row == ("vorticity",""):
            if field_col == ("streamfunction",""):
                mat = c1d.qid(res[0],0, bc, (kx**2 + (1/eta3**2)*ky**2))

            else:
                mat = c1d.zblk(res[0], bc)

        return mat

    def time_block(self, res, eq_params, eigs, bcs, field_row):
        """Create matrix block of time operator"""

        eta2 = np.sin(np.pi*eq_params['theta']/180)
        eta3 = np.cos(np.pi*eq_params['theta']/180)
        kx = eigs[0]
        ky = eigs[1]

        bc = self.convert_bc(eq_params,eigs,bcs,field_row,field_row)
        if field_row == ("streamfunction",""):
            mat = c1d.i1(res[0], bc, -(kx**2 + (1/eta3**2)*ky**2))

        elif field_row == ("velocityz",""):
            mat = c1d.i1(res[0], bc)

        elif field_row == ("temperature",""):
            mat = c1d.qid(res[0],0, bc)

            # Force temperature boundary condition
            if self.force_temperature_bc and not self.use_galerkin:
                mat = mat.tolil()
                mat[-2:,:] = 0
                mat = mat.tocsr()

        return mat
