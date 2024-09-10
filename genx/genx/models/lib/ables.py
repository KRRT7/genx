from functools import reduce

import numpy as np

from . import int_lay_xmean
from . import math_utils as mu

_ctype = np.complex128


def ass_P(k, d, ctype=_ctype):
    """Assemble the propagation matrix for the layers"""
    exp_term = np.exp(-1.0j * k * d)
    P = np.zeros((2, 2) + k.shape, dtype=ctype)
    P[0, 0] = exp_term
    P[1, 1] = 1.0 / exp_term
    return P


def ass_X(k, ctype=_ctype):
    """Assemble the interface matrix for all interfaces"""
    k_j = k[..., 1:]
    k_jm1 = k[..., :-1]

    X = np.empty((2, 2) + k_j.shape, dtype=ctype)

    X[0, 0] = (k_j + k_jm1) / k_jm1 / 2
    X[0, 1] = (k_jm1 - k_j) / k_jm1 / 2
    X[1, 0] = X[0, 1]
    X[1, 1] = X[0, 0]
    return X


def ass_X_2int(k_j, k_jm1, ctype=_ctype):
    """Assemble the interface matrix for all interfaces"""
    X = np.empty((2, 2) + k_j.shape, dtype=ctype)
    k_sum = k_j + k_jm1
    k_diff = k_jm1 - k_j
    k_inv = 1.0 / k_jm1

    X[0, 0] = k_sum * k_inv / 2
    X[0, 1] = k_diff * k_inv / 2
    X[1, 0] = X[0, 1]
    X[1, 1] = X[0, 0]
    return X


def ReflQ(Q, lamda, n, d, sigma):
    """Calculate the reflectivity from a multilayer stack with
    Abeles matrix formalism."""
    # pdb.set_trace()
    k_vac = 2 * np.pi / lamda
    kz = Q[:, np.newaxis] / 2.0
    n_amb2 = n[-1] * n[-1]
    k_j = np.sqrt((n * n - n_amb2) * k_vac * k_vac + n_amb2 * kz * kz)

    # print d.shape, n.shape, k_j.dtype
    X = ass_X(k_j)
    # X = im.ass_X_interfacelayer4(k_j, k_j, k_j, k_j*0, k_j*0, sigma, sigma*0, sigma*0)
    P = ass_P(k_j, d)

    PX = mu.dot2_Adiag(P[..., 1:-1], X[..., :-1])
    # print P.dtype, X.dtype, PX.dtype
    M = mu.dot2(X[..., -1], reduce(mu.dot2, np.rollaxis(PX, 3)[::-1]))

    r = M[1, 0] / M[0, 0]

    return np.abs(r) ** 2


def ReflQ_mag(Q, lamda, n, d, sigma, n_u, dd_u, sigma_u, n_l, dd_l, sigma_l):
    """Calculate the reflectivity from a multilayer stack with Abeles matrix formalism."""

    k_vac = 2 * np.pi / (lamda * np.ones(Q.shape))[:, np.newaxis]
    kz = Q[:, np.newaxis] / 2.0
    n_amb2 = n[:, 0] ** 2

    k = np.sqrt(
        (n**2 - n_amb2[:, np.newaxis]) * k_vac**2 + n_amb2[:, np.newaxis] * kz**2
    )
    k_l = np.sqrt(
        (n_l**2 - n_amb2[:, np.newaxis]) * k_vac**2 + n_amb2[:, np.newaxis] * kz**2
    )
    k_u = np.sqrt(
        (n_u**2 - n_amb2[:, np.newaxis]) * k_vac**2 + n_amb2[:, np.newaxis] * kz**2
    )

    X_lu = ass_X_2int(k_u[:, 1:], k_l[:, :-1])
    X_l = ass_X_2int(k_l[:, :-1], k[:, :-1])
    X_u = ass_X_2int(k[:, 1:], k_u[:, 1:])

    X = int_lay_xmean.calc_iso_Xmean(
        X_l, X_lu, X_u, k, k_l, k_u, dd_u, dd_l, sigma, sigma_l, sigma_u
    )
    P = ass_P(k, d)

    PX = mu.dot2_Adiag(P[..., 1:-1], X[..., 1:])
    M = mu.dot2(X[..., 0], reduce(mu.dot2, np.rollaxis(PX, 3)))

    r = M[1, 0] / M[0, 0]

    return np.abs(r) ** 2


def ReflQ_ref(Q, lamda, n, d, sigma):
    # Length of k-vector in vaccum
    d = d[1:-1]
    sigma = sigma[:-1]
    Q0 = 4 * np.pi / lamda
    # Calculates the wavevector in each layer
    Qj = np.sqrt((n[:, np.newaxis] ** 2 - n[-1] ** 2) * Q0**2 + n[-1] ** 2 * Q**2)
    # Fresnel reflectivity for the interfaces
    rp = (Qj[1:] - Qj[:-1]) / (
        Qj[1:] + Qj[:-1]
    )  # *exp(-Qj[1:]*Qj[:-1]/2*sigma[:,newaxis]**2)
    # print rp.shape #For debugging
    # print d.shape
    # print Qj[1:-1].shape
    p = np.exp(
        1.0j * d[:, np.newaxis] * Qj[1:-1]
    )  # Ignoring the top and bottom layer for the calc.
    # print p.shape #For debugging
    # Setting up a matrix for the reduce function. Reduce only takes one array
    # as argument
    rpp = np.array(list(map(lambda x, y: [x, y], rp[1:], p)))

    # print rpp.shape
    # Paratt's recursion formula
    def formula(rtot, rint):
        return (rint[0] + rtot * rint[1]) / (1 + rtot * rint[0] * rint[1])

    # Implement the recursion formula
    r = reduce(formula, rpp, rp[0])
    # print r.shape
    # return the reflectivity
    return np.abs(r) ** 2
