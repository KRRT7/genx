"""
Implementation of the iterative algorithm to define bi-layer periods of super-mirrors
that was introduced by J.B. Hayter and H.A. Mook in J. Appl. Cryst. (1989). 22, 35-41.

Adopted from Visual Basic script of PSI optics group.
"""

from numpy import arctan, log, pi, sqrt

SCALE_CONSTANT = 3.0 / sqrt(8.0)


def delta_tau(tau, delta_sign, rho_fraction, zeta):
    omega1 = sqrt(tau**2 - 1)
    omega2 = sqrt(tau**2 - rho_fraction)

    contrast = (omega2 - omega1) / (omega1 + omega2)
    kappa = (1 - contrast) / (1 + contrast)

    nu = log(1 - zeta) / (2 * log(kappa))

    if nu < 1:
        raise ValueError(
            "Nu should be below 1, check input parameters. "
            "This can be caused by a zeta value that is too small, try e.g. 0.95."
        )

    kappa_abs_1_nu = abs(kappa) ** (1 / nu)
    rho_bar = (1 - kappa_abs_1_nu) / (1 + kappa_abs_1_nu)
    arg = rho_bar * SCALE_CONSTANT

    sqrt_one_minus_arg2 = sqrt(1 - arg**2)
    d_omega1 = 2 * omega1 * arctan(arg / sqrt_one_minus_arg2) / pi
    wurzel = sqrt(1 + (omega1 + delta_sign * d_omega1) ** 2)

    return delta_sign * (wurzel - tau)


def sm_layers(rho1, rho2, N, zeta):
    tau = 1.1
    wert = SCALE_CONSTANT

    rho_fraction = rho2 / rho1

    D_SL = []
    delta_tau_negative_one = lambda tau: delta_tau(tau, -1, rho_fraction, zeta)

    for i in range(N):
        epsilon = 1.0
        while epsilon > 1e-4:
            temp1 = tau - delta_tau_negative_one(tau) - wert
            temp1_1_tau = 1.0001 * tau
            temp2 = (
                1e4
                * (temp1_1_tau - delta_tau_negative_one(temp1_1_tau) - wert - temp1)
                / tau
            )
            temp3 = temp1 / temp2
            tau -= temp3
            epsilon = abs(temp3 / tau)

        wert = tau + delta_tau(tau, 1, rho_fraction, zeta)

        omega1 = sqrt(tau**2 - 1)
        omega2 = sqrt(tau**2 - rho_fraction)

        sqrt_pi = sqrt(pi)
        sqrt_rho1 = sqrt(rho1)
        inv_4sqrt_rho1 = 1 / (4 * sqrt_rho1)

        d_1 = sqrt_pi * inv_4sqrt_rho1 / omega1
        d_2 = sqrt_pi * inv_4sqrt_rho1 / omega2

        D_SL.append([d_1, d_2])

    return D_SL
