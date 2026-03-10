"""
Direct HFNN Regressor - Mechanical Subsystem
=============================================

Resolves ω_{k+1} from discretized mechanical equation using HAM.

Implicit equation (Euler implícito):
    F_ω(ω_{k+1}) = ω_{k+1} - ω_k - T/J · [K_t·i_k - b·ω_{k+1} - N_load(ω_{k+1})] = 0

HAM corrections:
    ω̂_{k+1} = ω_k + ζ₁ + ζ₂
    ζ₁ = -F_ω(ω_k) / F'_ω(ω_k)          (Newton)
    ζ₂ = -(1/2)·F_ω(ω_k)²·F''_ω(ω_k) / F'_ω(ω_k)³  (Halley)

Author: Rodolfo H. Rodrigo
"""

import numpy as np


class DirectHFNN:
    """
    Direct HFNN regressor for mechanical subsystem.

    Parameters
    ----------
    K_t : float
        Torque constant [N·m/A]
    J : float
        Moment of inertia [kg·m²]
    b : float
        Viscous friction [N·m·s/rad]
    T : float
        Sampling period [s]
    """

    def __init__(self, K_t=0.1, J=0.001, b=0.01, T=0.001):
        self.K_t = K_t
        self.J = J
        self.b = b
        self.T = T

    def N_load(self, omega):
        """Load torque [N·m]."""
        c = 0.001  # Must match motor_model.py
        return c * omega**2

    def dN_load(self, omega):
        """First derivative of load torque."""
        c = 0.001
        return 2 * c * omega

    def d2N_load(self, omega):
        """Second derivative of load torque."""
        c = 0.001
        return 2 * c

    def predict(self, omega_k, i_k):
        """
        Predict ω_{k+1} using HFNN regressor.

        Parameters
        ----------
        omega_k : float or ndarray
            Current angular velocity [rad/s]
        i_k : float or ndarray
            Current current [A]

        Returns
        -------
        omega_k1 : float or ndarray
            Predicted angular velocity at k+1
        """
        # Evaluate F and derivatives at ω_k (initial guess)
        F = self._F_omega(omega_k, omega_k, i_k)
        F_prime = self._F_omega_prime(omega_k)
        F_double_prime = self._F_omega_double_prime(omega_k)

        # HAM corrections
        zeta1 = -F / F_prime
        zeta2 = -(0.5 * F**2 * F_double_prime) / F_prime**3

        omega_k1 = omega_k + zeta1 + zeta2
        return omega_k1

    def _F_omega(self, omega_k1, omega_k, i_k):
        """
        Implicit function F_ω(ω_{k+1}) = 0.

        F = ω_{k+1} - ω_k - T/J · [K_t·i_k - b·ω_{k+1} - N_load(ω_{k+1})]
        """
        tau = self.K_t * i_k - self.b * omega_k1 - self.N_load(omega_k1)
        return omega_k1 - omega_k - (self.T / self.J) * tau

    def _F_omega_prime(self, omega):
        """
        First derivative: dF/dω_{k+1}

        F'(ω) = 1 + (T/J)·[b + N'_load(ω)]
        """
        return 1.0 + (self.T / self.J) * (self.b + self.dN_load(omega))

    def _F_omega_double_prime(self, omega):
        """
        Second derivative: d²F/dω_{k+1}²

        F''(ω) = (T/J)·N''_load(ω)
        """
        return (self.T / self.J) * self.d2N_load(omega)

    def jacobian_wrt_state(self, omega_k, i_k):
        """
        Analytical Jacobian ∂ω_{k+1}/∂[ω_k, i_k] for EKF.

        Returns
        -------
        J : ndarray, shape (2,)
            [∂ω_{k+1}/∂ω_k, ∂ω_{k+1}/∂i_k]
        """
        # Using implicit function theorem on F(ω_{k+1}, ω_k, i_k) = 0
        F_prime = self._F_omega_prime(omega_k)

        # ∂ω_{k+1}/∂ω_k = -∂F/∂ω_k / ∂F/∂ω_{k+1}
        dF_domega_k = -1.0
        domega_k1_domega_k = -dF_domega_k / F_prime

        # ∂ω_{k+1}/∂i_k = -∂F/∂i_k / ∂F/∂ω_{k+1}
        dF_di_k = -(self.T / self.J) * self.K_t
        domega_k1_di_k = -dF_di_k / F_prime

        return np.array([domega_k1_domega_k, domega_k1_di_k])


if __name__ == '__main__':
    # Test regressor
    hfnn = DirectHFNN()

    omega_test = np.linspace(0, 100, 50)
    i_test = 3.0

    omega_pred = hfnn.predict(omega_test, i_test)

    print(f"Direct HFNN test:")
    print(f"  ω_k range: [{omega_test.min():.2f}, {omega_test.max():.2f}] rad/s")
    print(f"  i_k: {i_test:.2f} A")
    print(f"  ω_k+1 range: [{omega_pred.min():.2f}, {omega_pred.max():.2f}] rad/s")

    # Test Jacobian
    J = hfnn.jacobian_wrt_state(50.0, 3.0)
    print(f"\nJacobian at ω=50, i=3:")
    print(f"  ∂ω_k+1/∂ω_k = {J[0]:.6f}")
    print(f"  ∂ω_k+1/∂i_k = {J[1]:.6f}")
