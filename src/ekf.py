"""
Extended Kalman Filter — Continuous-Discrete Formulation
=========================================================

State: x = [ω, i]ᵀ
Measurement: z_k = ω_k + v_k  (Gaussian noise)

Prediction: integrates continuous dynamics with RK4
Jacobians: analytical from continuous-time model, discretized as F = I + A·T

The input u_k is unknown. For the EKF transition, we estimate u
from the electrical equation using the current state estimate.

After filtering, the reconstructed input is:
    û_k = R·i_k + L(i_k)·(i_k - i_{k-1})/T + K_e·ω_k

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import motor_model as mm


class EKF:
    """
    Continuous-discrete EKF for DC motor state estimation.

    Parameters
    ----------
    T : float
        Sampling period [s]
    Q : ndarray, shape (2, 2)
        Process noise covariance
    R_meas : float
        Measurement noise variance (on ω)
    x0 : ndarray, shape (2,)
        Initial state [ω₀, i₀]
    P0 : ndarray, shape (2, 2)
        Initial covariance
    """

    def __init__(self, T, Q, R_meas, x0, P0, use_integral=False):
        self.T = T
        self.Q = np.asarray(Q, dtype=float)
        self.R_meas = float(R_meas)
        self.H = np.array([[1.0, 0.0]])  # z = ω + noise

        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

        # History buffer for 4-point backward difference on i
        self.i_hist = [x0[1]] * 4  # [i_k, i_{k-1}, i_{k-2}, i_{k-3}]

        # For integral reconstruction
        self.use_integral = use_integral
        self.omega_prev = x0[0]

    def predict(self, u_est=0.0):
        """
        EKF prediction using RK4 transition + analytical Jacobian.

        Parameters
        ----------
        u_est : float
            Estimated input for transition (from previous step)
        """
        # State prediction via RK4
        self.x_pred = mm.rk4_step(self.x, u_est, self.T)

        # Analytical Jacobian (continuous) discretized: F ≈ I + A·T
        A = mm.jacobian_A_with_u(self.x, u_est)
        F = np.eye(2) + A * self.T

        # Covariance prediction
        self.P_pred = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        EKF measurement update.

        Parameters
        ----------
        z : float
            Measured ω (with Gaussian noise)

        Returns
        -------
        x : ndarray, shape (2,)
            Filtered state [ω̂, î]
        P : ndarray, shape (2, 2)
            Filtered covariance
        """
        # Innovation
        y = z - self.H @ self.x_pred

        # Innovation covariance
        S = self.H @ self.P_pred @ self.H.T + self.R_meas

        # Kalman gain
        K = self.P_pred @ self.H.T / S

        # State update
        self.x = self.x_pred + (K * y).flatten()

        # Covariance update (Joseph form)
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P_pred @ I_KH.T + (K * self.R_meas) @ K.T

        return self.x.copy(), self.P.copy()

    def reconstruct_input_derivative(self):
        """
        Reconstruct u_k via 4-point backward derivative (amplifies noise).

        di/dt ≈ (11·i_k - 18·i_{k-1} + 9·i_{k-2} - 2·i_{k-3}) / (6T)
        û_k = R·i_k + L(i_k)·di/dt + K_e·ω_k
        """
        omega_k, i_k = self.x
        L_ik = mm.L_sat(i_k)

        # Shift history and insert new i_k
        self.i_hist = [i_k] + self.i_hist[:3]

        di_dt = (11*self.i_hist[0] - 18*self.i_hist[1]
                 + 9*self.i_hist[2] - 2*self.i_hist[3]) / (6 * self.T)

        u_k = mm.R * i_k + L_ik * di_dt + mm.K_e * omega_k
        return u_k

    def reconstruct_input_integral(self):
        """
        Reconstruct u_k via integral formulation (smooths noise).

        Phi(i) = L0·I_SAT·arctan(i/I_SAT)
        u_k = [Phi(i_k) - Phi(i_{k-1})]/T + R·(i_{k-1}+i_k)/2 + K_e·(ω_{k-1}+ω_k)/2
        """
        omega_k, i_k = self.x

        # Shift history
        i_prev = self.i_hist[0]
        omega_prev = self.omega_prev
        self.i_hist = [i_k] + self.i_hist[:3]
        self.omega_prev = omega_k

        Phi_k = mm.L0 * mm.I_SAT * np.arctan(i_k / mm.I_SAT)
        Phi_prev = mm.L0 * mm.I_SAT * np.arctan(i_prev / mm.I_SAT)

        u_k = (Phi_k - Phi_prev) / self.T \
            + mm.R * (i_prev + i_k) / 2.0 \
            + mm.K_e * (omega_prev + omega_k) / 2.0
        return u_k

    def reconstruct_input(self):
        """Reconstruct u_k using the selected method."""
        if self.use_integral:
            return self.reconstruct_input_integral()
        else:
            return self.reconstruct_input_derivative()

    def step(self, z, u_est=0.0):
        """
        Complete EKF step: predict → update → reconstruct input.

        Parameters
        ----------
        z : float
            Measurement (noisy ω)
        u_est : float
            Estimated input for prediction

        Returns
        -------
        x_filt : ndarray, shape (2,)
            Filtered [ω̂, î]
        P_filt : ndarray, shape (2, 2)
        u_hat : float
            Reconstructed input
        """
        self.predict(u_est)
        x_filt, P_filt = self.update(z)
        u_hat = self.reconstruct_input()
        return x_filt, P_filt, u_hat
