from dataclasses import dataclass, field

from operators.polynomials import *
from utils import Timer


@dataclass
class AssociatedLegendreTransformSingleM(ABC):
    """
    The transforms of associated Legendre functions from spectral space to physical space
    """
    maxnl: int
    m: int
    grid: np.ndarray = field(repr=False)

    def __post_init__(self):
        self._operators = dict()
        self._operators['plm'] = Plm(self.m, self.maxnl - 1, self.grid)
        self._operators['plmdivsin'] = PlmDivSin(self.m, self.maxnl, self.grid)
        self._operators['dthetaplm'] = DthetaPlm(self.m, self.maxnl - 1, self._operators['plm'],
                                                 self._operators['plmdivsin'], self.grid)
        self._operators['plmdivsin'] = self._operators['plmdivsin'][:, :-1]
        c_t, s_t = np.cos(self.grid).reshape(-1, 1), np.sin(self.grid).reshape(-1, 1)
        l_factor = np.array([l*(l + 1) for l in range(self.m, self.maxnl)])
        self._operators['dtheta2plm'] = (self.m**2*self._operators['plmdivsin'] - c_t*self._operators['dthetaplm'])/s_t \
            - l_factor*self._operators['plm']

    @property
    def operators(self):
        return self._operators


if __name__ == "__main__":
    with Timer("init op"):
        transform = AssociatedLegendreTransformSingleM(41, 1, np.linspace(0, np.pi, 501))
