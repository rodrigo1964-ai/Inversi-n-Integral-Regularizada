"""
DC Motor Model with Nonlinear Magnetic Saturation
===================================================

Ground truth simulation for dual HFNN + Kalman paper.

Motor equations (continuous time):
- Mechanical: J·dω/dt = K_t·i - b·ω - N_load(ω)
- Electrical: L(i)·di/dt = u - R·i - K_e·ω

State: x = [ω, i]ᵀ
Input: u (voltage, unknown to estimator)

L(i) = L0 / (1 + (i/i_sat)²) — known saturation curve.

Author: Rodolfo H. Rodrigo
"""

import numpy as np


# Motor parameters (all known)
R = 1.0         # Ohm, armature resistance
K_e = 0.1       # V·s/rad, back-EMF constant
K_t = 0.1       # N·m/A, torque constant
J = 0.001       # kg·m², moment of inertia
b = 0.01        # N·m·s/rad, viscous friction
L0 = 0.01       # H, unsaturated inductance
I_SAT = 10.0    # A, saturation current
C_LOAD = 1e-5   # Load torque coefficient


def L_sat(i):
    """L(i) = L0 / (1 + (i/i_sat)²)"""
    return L0 / (1.0 + (i / I_SAT)**2)


def dL_di(i):
    """dL/di"""
    d = 1.0 + (i / I_SAT)**2
    return -2.0 * L0 * i / (I_SAT**2 * d**2)


def d2L_di2(i):
    """d²L/di²"""
    d = 1.0 + (i / I_SAT)**2
    return (-2.0 * L0 / I_SAT**2 + 8.0 * L0 * i**2 / (I_SAT**4 * d)) / d**2


def N_load(omega):
    """Load torque: N_load = c·ω²"""
    return C_LOAD * omega**2


def dN_load(omega):
    """dN_load/dω = 2c·ω"""
    return 2.0 * C_LOAD * omega


def dynamics(x, u):
    """
    Continuous-time dynamics: dx/dt = f(x, u)

    Parameters
    ----------
    x : ndarray, shape (2,)
        State [ω, i]
    u : float
        Input voltage [V]

    Returns
    -------
    dxdt : ndarray, shape (2,)
        [dω/dt, di/dt]
    """
    omega, i = x

    d_omega = (K_t * i - b * omega - N_load(omega)) / J
    d_i = (u - R * i - K_e * omega) / L_sat(i)

    return np.array([d_omega, d_i])


def jacobian_A(x):
    """
    Analytical Jacobian A = ∂f/∂x of continuous-time dynamics.

    A = [[∂f₁/∂ω, ∂f₁/∂i],
         [∂f₂/∂ω, ∂f₂/∂i]]

    Parameters
    ----------
    x : ndarray, shape (2,)
        State [ω, i]

    Returns
    -------
    A : ndarray, shape (2, 2)
    """
    omega, i = x
    L_i = L_sat(i)
    dL = dL_di(i)

    # ∂(dω/dt)/∂ω = -(b + dN_load/dω) / J
    a11 = -(b + dN_load(omega)) / J

    # ∂(dω/dt)/∂i = K_t / J
    a12 = K_t / J

    # ∂(di/dt)/∂ω = -K_e / L(i)
    a21 = -K_e / L_i

    # ∂(di/dt)/∂i = [-R·L(i) - (u - R·i - K_e·ω)·L'(i)] / L(i)²
    # = -R/L(i) - (di/dt)·L'(i)/L(i)
    di_dt = (0.0 - R * i - K_e * omega) / L_i  # Note: u unknown, use 0 for Jacobian linearization
    a22 = -R / L_i - di_dt * dL / L_i

    return np.array([[a11, a12], [a21, a22]])


def jacobian_A_with_u(x, u):
    """
    Analytical Jacobian A = ∂f/∂x with known u (for ground truth validation).
    """
    omega, i = x
    L_i = L_sat(i)
    dL = dL_di(i)

    a11 = -(b + dN_load(omega)) / J
    a12 = K_t / J
    a21 = -K_e / L_i

    di_dt = (u - R * i - K_e * omega) / L_i
    a22 = -R / L_i - di_dt * dL / L_i

    return np.array([[a11, a12], [a21, a22]])


def rk4_step(x, u, dt):
    """Single RK4 integration step."""
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt/2 * k1, u)
    k3 = dynamics(x + dt/2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(t_span, x0, u_func, dt_rk4=1e-5):
    """
    Simulate motor with RK4 (fine resolution ground truth).

    Parameters
    ----------
    t_span : tuple (t0, tf)
    x0 : ndarray, shape (2,)
        Initial [ω₀, i₀]
    u_func : callable
        u(t) → voltage
    dt_rk4 : float
        RK4 integration step

    Returns
    -------
    t, states, inputs : ndarrays
    """
    t0, tf = t_span
    t = np.arange(t0, tf, dt_rk4)
    n = len(t)

    states = np.zeros((n, 2))
    inputs = np.zeros(n)
    states[0] = x0

    for k in range(n - 1):
        u_k = u_func(t[k])
        inputs[k] = u_k
        states[k+1] = rk4_step(states[k], u_k, dt_rk4)

    inputs[-1] = u_func(t[-1])
    return t, states, inputs


def generate_experiment_data(T_sample=0.0001, n_samples=2000,
                             u_type='step', noise_std=0.1, seed=42):
    """
    Generate experiment data: ground truth + Gaussian noisy measurements.

    Parameters
    ----------
    T_sample : float
        Sampling period [s]
    n_samples : int
        Number of measurement samples
    u_type : str
        'step', 'ramp', 'pulse'
    noise_std : float
        Std dev of Gaussian noise on ω [rad/s]
    seed : int
        Random seed

    Returns
    -------
    t_s : ndarray
        Sample times
    omega_true, omega_meas, i_true, u_samples : ndarrays
    """
    np.random.seed(seed)

    t_total = n_samples * T_sample

    if u_type == 'step':
        u_func = lambda t: 12.0 if t > 0.005 else 0.0
    elif u_type == 'ramp':
        u_func = lambda t: min(12.0, 600.0 * t)
    elif u_type == 'pulse':
        u_func = lambda t: 12.0 * (np.sin(2 * np.pi * 50 * t) > 0)
    else:
        raise ValueError(f"Unknown u_type: {u_type}")

    # Fine-resolution ground truth
    x0 = np.array([0.0, 0.0])
    t, states, inputs = simulate((0, t_total), x0, u_func, dt_rk4=1e-6)

    # Downsample
    step = int(T_sample / 1e-6)
    indices = np.arange(0, len(t), step)[:n_samples]

    t_s = t[indices]
    omega_true = states[indices, 0]
    i_true = states[indices, 1]
    u_samples = inputs[indices]

    # Gaussian measurement noise on ω only
    omega_meas = omega_true + np.random.normal(0, noise_std, len(omega_true))

    return t_s, omega_true, omega_meas, i_true, u_samples


if __name__ == '__main__':
    t, omega_true, omega_meas, i_true, u = generate_experiment_data(
        n_samples=2000, u_type='step', noise_std=0.1
    )

    print(f"ω range: [{omega_true.min():.3f}, {omega_true.max():.3f}] rad/s")
    print(f"i range: [{i_true.min():.3f}, {i_true.max():.3f}] A")
    print(f"u range: [{u.min():.1f}, {u.max():.1f}] V")
    print(f"Δω max: {np.max(np.abs(np.diff(omega_true))):.6f}")
    print(f"Amplification J/(T·K_t) = {J/(0.0001*K_t):.1f}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ms = t * 1000
    axes[0].plot(ms, u, 'k-')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].grid(True)

    axes[1].plot(ms, omega_true, 'b-', label='True')
    axes[1].plot(ms, omega_meas, 'r.', alpha=0.3, markersize=1, label='Measured')
    axes[1].set_ylabel('$\\omega$ [rad/s]')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(ms, i_true, 'g-')
    axes[2].set_ylabel('$i$ [A]')
    axes[2].set_xlabel('Time [ms]')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/motor_ground_truth.png', dpi=300)
    print("Saved to results/motor_ground_truth.png")
