from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse.linalg as spla
from typing import Union, Dict

from operators.equations import *
from operators.worland_transform import WorlandTransform
from utils import *


class BaseModel(ABC):
    def __init__(self, nr, maxnl, m, n_grid, *args, **kwargs):
        self.res = (nr, maxnl, m)
        if n_grid is None:
            n_grid = nr + maxnl//2 + 11
        self.n_grid = n_grid

    @abstractmethod
    def setup_operator(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup_eigen_problem(self, operators, **kwargs):
        pass


class FreeDecay:
    def __init__(self, component, nr, l):
        bcs = {'tor': {0: 10}, 'pol': {0: 13}}
        self.bc = bcs[component]
        self.l = l
        self.nr = nr
        self.component = component

    def setup_eigen_problem(self):
        import quicc.geometry.spherical.sphere_radius_worland as rad
        A = rad.i2lapl(self.nr, self.l, self.bc, coeff=self.l*(self.l+1))
        B = rad.i2(self.nr, self.l, {0: 0}, coeff=self.l*(self.l+1))
        return A, B


class KinematicDynamo(BaseModel):
    def __init__(self, nr, maxnl, m, n_grid=None, **kwargs):
        super(KinematicDynamo, self).__init__(nr, maxnl, m, n_grid)
        self.transform = WorlandTransform(nr, maxnl, m, self.n_grid, require_curl=False)
        self.eq_setup = kwargs
        self.induction_eq = InductionEquation(nr, maxnl, m, **self.eq_setup)

    def setup_operator(self, flow_modes: List[SphericalHarmonicMode], setup_eigen=False, **kwargs):
        induction_mat = self.induction_eq.induction(self.transform, flow_modes, imposed_flow=True, quasi_inverse=True)
        operators = {'mass': self.induction_eq.mass,
                     'induction': induction_mat,
                     'diffusion': self.induction_eq.diffusion}
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        Rm = kwargs.get('Rm')
        return Rm * operators['induction'] + operators['diffusion'], operators['mass']


class InertialModes(BaseModel):
    def __init__(self, nr, maxnl, m, inviscid: bool, bc_type: str = None):
        super(InertialModes, self).__init__(nr, maxnl, m, None)
        self.momentum_eq = MomentumEquation(*self.res, inviscid, bc_type)

    def setup_operator(self, setup_eigen=False, **kwargs):
        nr, maxnl, m = self.res
        dim = nr * (maxnl - m)
        operators = {}
        operators['mass'] = self.momentum_eq.mass
        operators['coriolis'] = self.momentum_eq.coriolis
        if self.momentum_eq.inviscid:
            operators['diffusion'] = scsp.csc_matrix((2 * dim, 2 * dim))
        else:
            operators['diffusion'] = self.momentum_eq.diffusion
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        if not self.momentum_eq.inviscid:
            ekman = kwargs.get('ekman')
            return -2*operators['coriolis'] + ekman*operators['diffusion'], operators['mass']
        else:
            return -2*operators['coriolis'], operators['mass']


class MagnetoCoriolis(BaseModel):
    def __init__(self, nr, maxnl, m, n_grid=None, inviscid=True, bc=None, **kwargs):
        super(MagnetoCoriolis, self).__init__(nr, maxnl, m, n_grid)
        self.transform = WorlandTransform(nr, maxnl, m, self.n_grid, require_curl=True)
        self.inviscid = inviscid
        self.induction_eq_setup = kwargs
        if not inviscid:
            assert bc is not None
            self.bc = bc
        self.induction_eq = InductionEquation(*self.res, **self.induction_eq_setup)
        self.momentum_eq = MomentumEquation(*self.res, inviscid=inviscid, bc_type=bc)
        if self.induction_eq.galerkin:
            self.dim = {'u': 2*nr*(maxnl-m), 'b': 2*(nr-1)*(maxnl-m)}
        else:
            self.dim = {'u': 2*nr*(maxnl-m), 'b': 2*nr*(maxnl - m)}

    def setup_operator(self, field_modes: List[SphericalHarmonicMode],
                       flow_modes: Union[None, List[SphericalHarmonicMode]] = None, setup_eigen=False,
                       *args, **kwargs):
        if flow_modes is None:
            flow_modes = []
        nr, maxnl, m = self.res
        dim = nr*(maxnl - m)
        operators = {}
        operators['lorentz'] = self.momentum_eq.lorentz(self.transform, field_modes, quasi_inverse=True)
        if self.induction_eq.galerkin:
            operators['lorentz'] = operators['lorentz'] @ self.induction_eq.stencil
        operators['inductionB'] = self.induction_eq.induction(self.transform, field_modes, imposed_flow=False,
                                                              quasi_inverse=True)
        operators['advection'] = self.momentum_eq.advection(self.transform, flow_modes, quasi_inverse=True)
        operators['inductionU'] = self.induction_eq.induction(self.transform, flow_modes, imposed_flow=True,
                                                              quasi_inverse=True)
        operators['magnetic_diffusion'] = self.induction_eq.diffusion
        operators['coriolis'] = self.momentum_eq.coriolis
        if self.inviscid:
            operators['viscous_diffusion'] = scsp.csr_matrix((2*dim, 2*dim))
        else:
            operators['viscous_diffusion'] = self.momentum_eq.diffusion
        operators['induction_mass'] = self.induction_eq.mass
        operators['momentum_mass'] = self.momentum_eq.mass
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        Eeta = kwargs.get('magnetic_ekman')
        elsasser = kwargs.get('elsasser')
        U = kwargs.get('U', 0)
        ekman = kwargs.get('ekman', 0)

        if Eeta == 0:
            clu = spla.splu(operators['coriolis'])
            operators['inv_coriolis'] = clu
            u = clu.solve(operators['lorentz'].toarray())
            operators['ms_induction'] = operators['inductionB'] @ u

            B = operators['induction_mass']
            A = elsasser*operators['ms_induction'] + operators['magnetic_diffusion']
            # separate parity
            if kwargs.get('parity', False):
                return self.separate_parity(A, B, b_parity='DP', u_parity=None), \
                       self.separate_parity(A, B, b_parity='QP', u_parity=None)
            else:
                return A, B
        else:
            B = scsp.block_diag((Eeta*operators['momentum_mass'], operators['induction_mass']))
            A = scsp.bmat([[-Eeta*U*operators['advection']-operators['coriolis'] + ekman*operators['viscous_diffusion'],
                            elsasser**0.5*operators['lorentz']],
                           [elsasser**0.5*operators['inductionB'],
                            U*operators['inductionU'] + operators['magnetic_diffusion']]
                           ])
            # separate parity
            if kwargs.get('parity', False):
                return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))),\
                       self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
            else:
                return A, B

    def separate_parity(self, A, B, b_parity, u_parity):
        nr, maxnl, m = self.res
        dimu = 2 * nr * (maxnl-m)
        A = scsp.lil_matrix(A)
        B = scsp.lil_matrix(B)

        if u_parity is None:
            row_idx = vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin))
            col_idx = vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin))
        else:
            row_idx = np.append(vector_parity_idx(nr, maxnl, m, u_parity),
                                dimu + vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin)))
            col_idx = np.append(vector_parity_idx(nr, maxnl, m, u_parity),
                                dimu + vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin)))
        return scsp.csr_matrix(A[row_idx[:, None], col_idx]), scsp.coo_matrix(B[row_idx[:, None], col_idx])

    def u_parity(self, b_parity, relation):
        if relation == 'same':
            return b_parity
        else:
            return 'DP' if b_parity == 'QP' else 'QP'


class IdealMagnetoCoriolis(MagnetoCoriolis):
    """ Ideal Magneto-Coriolis modes, using the Alfven time scale formulation with Le number """
    def __init__(self, nr, maxnl, m, n_grid=None, **kwargs):
        # default using a galerkin basis, but can be used with no boundary condition
        super(IdealMagnetoCoriolis, self).__init__(nr, maxnl, m, n_grid, inviscid=True,
                                                   galerkin=kwargs.get('galerkin', True),
                                                   ideal=True,
                                                   boundary_condition=kwargs.get('boundary_condition', True))

    def setup_eigen_problem(self, operators, **kwargs):
        Le = kwargs.get('lehnert')
        U = kwargs.get('U', 0)
        B = scsp.block_diag((operators['momentum_mass'], operators['induction_mass']))
        A = scsp.bmat(
            [[-U * operators['advection'] - 2/Le * operators['coriolis'], operators['lorentz']],
             [operators['inductionB'], U * operators['inductionU']]
             ])
        # separate parity
        if kwargs.get('parity', False):
            return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))), \
                   self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
        else:
            return A, B


class TorsionalOscillation(MagnetoCoriolis):
    """ torsional oscillation with magnetic diffusion and possibly with viscous diffusion
        Non-dimensional parameters are Le, Lu and Pm """
    def __init__(self, nr, maxnl, inviscid=True, n_grid=None, **kwargs):
        super(TorsionalOscillation, self).__init__(nr, maxnl, 0, n_grid, inviscid=inviscid,
                                                   galerkin=kwargs.get('galerkin', True))

    def setup_eigen_problem(self, operators, **kwargs):
        Le = kwargs.get('lehnert')
        Lu = kwargs.get('lundquist')
        Pm = kwargs.get('pm', 0)
        U = kwargs.get('U', 0)
        B = scsp.block_diag((operators['momentum_mass'], operators['induction_mass']))
        A = scsp.bmat(
            [[-U * operators['advection'] - 2 / Le * operators['coriolis'] + Pm/Lu*operators['viscous_diffusion'],
              operators['lorentz']],
             [operators['inductionB'], U * operators['inductionU'] + 1/Lu*operators['magnetic_diffusion']]
             ])
        # separate parity
        if kwargs.get('parity', False):
            return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))), \
                   self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
        else:
            return A, B


class IdealTorsionalOscillation(IdealMagnetoCoriolis):
    """ torsional oscillation with no diffusion, galerkin basis """
    def __init__(self, nr, maxnl, n_grid=None, **kwargs):
        super(IdealTorsionalOscillation, self).__init__(nr, maxnl, 0, n_grid, **kwargs)

    def setup_eigen_problem(self, operators, **kwargs):
        Le = kwargs.get('lehnert')
        U = kwargs.get('U', 0)
        B = scsp.block_diag((operators['momentum_mass'], operators['induction_mass']))
        A = scsp.bmat(
            [[-U * operators['advection'] - 2 / Le * operators['coriolis'],
              operators['lorentz']],
             [operators['inductionB'], U * operators['inductionU']]
             ])
        # separate parity
        if kwargs.get('parity', False):
            return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))), \
                   self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
        else:
            return A, B
