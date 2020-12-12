"""
The Python script abstract_dmd.py provides the default construct for all further DMD scripts.
"""

import torch as tr


class AbstractDMD(object):
    def __init__(self):
        self.__mu: tr.Tensor = tr.Tensor()  # DMD-Eigenvalues
        self.__phi: tr.Tensor = tr.Tensor()  # DMD-Modes

    @property
    def mu(self) -> tr.Tensor:
        """
        Getter function of DMD-Eigenvalue
        :return: Torch Tensor
        """
        return self.__mu

    @mu.setter
    def mu(self, mu: tr.Tensor):
        """
        Setter function of DMD-Eigenvalue
        :return: None
        """
        self.__mu = mu

    @property
    def phi(self) -> tr.Tensor:
        """
        Setter function of DMD-Mode
        :return:Torch Tensor
        """
        return self.__phi

    @phi.setter
    def phi(self, phi: tr.Tensor):
        """
        Setter function of DMD-Mode
        :return: None
        """
        self.__phi = phi

    def fit(self, Data: tr.Tensor, truncation: int = None):
        raise NotImplementedError

    def predict(self, t_steps: tr.Tensor):
        raise NotImplementedError
