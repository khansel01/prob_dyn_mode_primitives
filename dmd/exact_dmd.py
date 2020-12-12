"""
An implementation of exact dynamic mode decomposition based on Jonathan H. Tu et al.
On dynamic mode decomposition: Theory and applications, arXiv preprint arXiv:1312.0041, 2013
"""

import torch as tr
from dmd.abstract_dmd import AbstractDMD


class ExactDMD(AbstractDMD):
    def __init__(self):
        super().__init__()
        pass

    # override
    def fit(self, Data: tr.Tensor, truncation: int = None):
        """
        Fit Parameters to Data using Exact-Dynamic Mode Decomposition
        :param Data: A 2xNxT Tensor where the first two dimensions correspond to X0 and X1
        :param truncation: Select the truncation of the Singular Value Decomposition
        :return: None
        """
        X0, X1 = Data[0].type(tr.complex64), Data[1].type(tr.complex64)

        U, S, V = tr.svd(X0)
        if truncation is None:
            truncation = S.cumsum(dim=0)/S.sum() <= 0.9999

        U_r = U[:, truncation].type(tr.complex64)
        S_r = S[truncation].diag().type(tr.complex64)
        V_r = V.conj()[:, truncation].type(tr.complex64)
        print(V.shape)

        A_tilde = U_r.conj().T @ X1 @ V_r @ tr.inverse(S_r)
        mu_tilde, phi_tilde = tr.eig(A_tilde.real, eigenvectors=True)
        self.mu = tr.complex(mu_tilde[:, 0], mu_tilde[:, 1])

        mu_diag = tr.complex(self.mu.real.diag(), self.mu.imag.diag()).inverse()

        _phi_tilde = tr.zeros_like(phi_tilde, dtype=tr.complex64)
        _phi_tilde[:, 0] = tr.complex(phi_tilde[:, 0], phi_tilde[:, 1])
        _phi_tilde[:, 1] = tr.complex(phi_tilde[:, 0], - phi_tilde[:, 1])
        _phi_tilde[:, 2] = tr.complex(phi_tilde[:, 2], tr.zeros_like(phi_tilde[:, 2]))

        self.phi = X1 @ V_r @ S_r.inverse() @ _phi_tilde @ mu_diag
        self.b = tr.inverse(self.phi.T @ self.phi) @ self.phi.T @ X0[:, 0]



        import numpy as np
        from numpy.linalg import svd, eig, inv

        X, Y = X0.numpy(), X1.numpy()

        u, s, v = svd(X, full_matrices=False)

        if truncation is None:
            truncation = s.cumsum()/s.sum() <= 0.9999

        ur = u[:, truncation]
        Sig = np.diag(s[truncation])
        vr = v.conj().T[:, truncation]
        print(v.shape)
        # A_tilde
        atilde_n = ur.conj().T @ Y @ vr @ inv(Sig)
        # DMD Mode
        mu_n, W_n = eig(atilde_n)
        # X Mode

        phi_ = Y @ vr @ inv(Sig) @ W_n @ np.diag(1 / mu_n)
        b = np.linalg.lstsq(phi_, X[:, 0], rcond=-1)[0]

        phi_tilde = tr.tensor(W_n)

    # override
    def predict(self, t_steps: tr.Tensor):
        b_diag = tr.complex(self.b.real.diag(), self.b.imag.diag())
        return self.phi @ (b_diag) @ tr.vander(self.mu, len(t_steps), increasing=True)
