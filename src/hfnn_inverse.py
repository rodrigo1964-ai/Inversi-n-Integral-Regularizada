"""
Inverse HFNN — Input Reconstruction from Filtered State
=========================================================

Given filtered state estimates [ω̂_k, ω̂_{k-1}] from the EKF:

1. Extract current i_{k-1} from the mechanical equation (algebraic):
   i_{k-1} = [J·(ω_k - ω_{k-1})/T + b·ω_k + N_load(ω_k)] / K_t

2. Reconstruct input u_{k-1} from the electrical equation:
   u_{k-1} = R·i_{k-1} + L(i_{k-1})·(i_{k-1} - i_{k-2})/T + K_e·ω_{k-1}

Both steps are algebraically explicit. L(i) is the known saturation curve.

Author: Rodolfo H. Rodrigo
"""

import numpy as np


class InverseHFNN:
    """
    Input reconstruction from filtered state estimates.

    Parameters
    ----------
    R : float
        Armature resistance [Ohm]
    K_e : float
        Back-EMF constant [V·s/rad]
    K_t : float
        Torque constant [N·m/A]
    J : float
        Moment of inertia [kg·m²]
    b : float
        Viscous friction [N·m·s/rad]
    L0 : float
        Unsaturated inductance [H]
    i_sat : float
        Saturation current [A]
    T : float
        Sampling period [s]
    """

    def __init__(self, R=1.0, K_e=0.1, K_t=0.1, J=0.001, b=0.01,
                 L0=0.01, i_sat=10.0, T=0.001):
        self.R = R
        self.K_e = K_e
        self.K_t = K_t
        self.J = J
        self.b = b
        self.L0 = L0
        self.i_sat = i_sat
        self.T = T

    def L_sat(self, i):
        """Known saturation curve: L(i) = L0 / (1 + (i/i_sat)²)"""
        return self.L0 / (1.0 + (i / self.i_sat)**2)

    def dL_di(self, i):
        """First derivative of L(i) w.r.t. i."""
        d = 1.0 + (i / self.i_sat)**2
        return -2.0 * self.L0 * i / (self.i_sat**2 * d**2)

    def N_load(self, omega):
        """Load torque: N_load = c·ω²"""
        c = 0.001
        return c * omega**2

    def extract_current(self, omega_k, omega_k_1):
        """
        Extract i_{k-1} from the discretized mechanical equation.

        From backward Euler:
            J·(ω_k - ω_{k-1})/T = K_t·i_{k-1} - b·ω_k - N_load(ω_k)

        Solving for i_{k-1}:
            i_{k-1} = [J·(ω_k - ω_{k-1})/T + b·ω_k + N_load(ω_k)] / K_t

        Parameters
        ----------
        omega_k : float
            Angular velocity at time k
        omega_k_1 : float
            Angular velocity at time k-1

        Returns
        -------
        i_k_1 : float
            Estimated current at time k-1
        """
        numerator = (self.J * (omega_k - omega_k_1) / self.T
                     + self.b * omega_k
                     + self.N_load(omega_k))
        return numerator / self.K_t

    def reconstruct_input(self, i_k, i_k_1, omega_k):
        """
        Reconstruct u_k from the discretized electrical equation.

        From backward Euler:
            u_k = R·i_k + L(i_k)·(i_k - i_{k-1})/T + K_e·ω_k

        Parameters
        ----------
        i_k : float
            Current at time k
        i_k_1 : float
            Current at time k-1
        omega_k : float
            Angular velocity at time k

        Returns
        -------
        u_k : float
            Reconstructed input voltage
        """
        L_ik = self.L_sat(i_k)
        u_k = self.R * i_k + L_ik * (i_k - i_k_1) / self.T + self.K_e * omega_k
        return u_k

    def jacobian_current_wrt_omega(self):
        """
        Analytical Jacobian ∂i_{k-1}/∂[ω_k, ω_{k-1}] for uncertainty propagation.

        ∂i/∂ω_k   = [J/T + b + N'_load(ω_k)] / K_t
        ∂i/∂ω_{k-1} = -J/(T·K_t)

        Note: N'_load depends on ω_k, so this returns a function.
        """
        def jac(omega_k):
            di_domega_k = (self.J / self.T + self.b + 2 * 0.001 * omega_k) / self.K_t
            di_domega_k_1 = -self.J / (self.T * self.K_t)
            return np.array([di_domega_k, di_domega_k_1])
        return jac

    def jacobian_input_wrt_current(self, i_k, i_k_1):
        """
        Analytical Jacobian ∂u_k/∂[i_k, i_{k-1}] for uncertainty propagation.

        ∂u/∂i_k   = R + L'(i_k)·(i_k - i_{k-1})/T + L(i_k)/T
        ∂u/∂i_{k-1} = -L(i_k)/T
        """
        L_ik = self.L_sat(i_k)
        dL = self.dL_di(i_k)

        du_di_k = self.R + dL * (i_k - i_k_1) / self.T + L_ik / self.T
        du_di_k_1 = -L_ik / self.T

        return np.array([du_di_k, du_di_k_1])


if __name__ == '__main__':
    inv = InverseHFNN()

    # Test with known values
    omega_k = 10.0
    omega_k_1 = 9.5
    i_est = inv.extract_current(omega_k, omega_k_1)
    print(f"Extracted current: i = {i_est:.4f} A")

    i_k_1 = i_est * 0.95
    u_est = inv.reconstruct_input(i_est, i_k_1, omega_k)
    print(f"Reconstructed input: u = {u_est:.4f} V")

    jac_i = inv.jacobian_current_wrt_omega()(omega_k)
    print(f"∂i/∂[ω_k, ω_{{k-1}}] = {jac_i}")

    jac_u = inv.jacobian_input_wrt_current(i_est, i_k_1)
    print(f"∂u/∂[i_k, i_{{k-1}}] = {jac_u}")
