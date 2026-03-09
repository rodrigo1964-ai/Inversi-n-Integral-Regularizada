"""
Test inverso: reconstruir u(t) a partir de ω(t) e i(t) de RK4.

Ecuación eléctrica:
    u = L(i)·di/dt + R·i + K_e·ω

di/dt se aproxima con diferencias finitas backward:
    2 puntos: (i_k - i_{k-1}) / T
    3 puntos: (3·i_k - 4·i_{k-1} + i_{k-2}) / (2T)
    4 puntos: (11·i_k - 18·i_{k-1} + 9·i_{k-2} - 2·i_{k-3}) / (6T)

Sin ruido. Compara û vs u verdadera.

Author: Rodolfo H. Rodrigo
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import motor_model as mm


def reconstruct_u(omega, i_arr, T, n_points=3):
    """
    Reconstruct u from known ω and i using electrical equation.

    Parameters
    ----------
    omega : ndarray
        Angular velocity [rad/s]
    i_arr : ndarray
        Current [A]
    T : float
        Sampling period [s]
    n_points : int
        Number of backward points for di/dt (2, 3, or 4)

    Returns
    -------
    u_hat : ndarray
        Reconstructed input voltage
    """
    n = len(omega)
    u_hat = np.zeros(n)

    for k in range(n_points - 1, n):
        i_k = i_arr[k]
        L_ik = mm.L_sat(i_k)

        if n_points == 2:
            di_dt = (i_arr[k] - i_arr[k-1]) / T
        elif n_points == 3:
            di_dt = (3*i_arr[k] - 4*i_arr[k-1] + i_arr[k-2]) / (2*T)
        elif n_points == 4:
            di_dt = (11*i_arr[k] - 18*i_arr[k-1] + 9*i_arr[k-2] - 2*i_arr[k-3]) / (6*T)
        else:
            raise ValueError(f"n_points must be 2, 3, or 4")

        u_hat[k] = L_ik * di_dt + mm.R * i_k + mm.K_e * omega[k]

    return u_hat


def run_test(T=0.0001, t_final=0.2):
    """Test inverse regressor against RK4 ground truth."""

    # Ground truth RK4
    u_func = lambda t: 12.0 if t > 0.005 else 0.0
    x0 = np.array([0.0, 0.0])
    t_rk4, states_rk4, inputs_rk4 = mm.simulate(
        (0, t_final), x0, u_func, dt_rk4=1e-6
    )

    # Downsample
    n = int(t_final / T)
    step = int(T / 1e-6)
    idx = np.arange(0, len(t_rk4), step)[:n]

    t_ref = t_rk4[idx]
    omega_ref = states_rk4[idx, 0]
    i_ref = states_rk4[idx, 1]
    u_ref = inputs_rk4[idx]

    # Reconstruir u con 2, 3, y 4 puntos
    results = {}
    for np_ in [2, 3, 4]:
        u_hat = reconstruct_u(omega_ref, i_ref, T, n_points=np_)
        skip = max(np_, int(0.01 / T))  # skip step transient
        err_max = np.max(np.abs(u_hat[skip:] - u_ref[skip:]))
        err_rmse = np.sqrt(np.mean((u_hat[skip:] - u_ref[skip:])**2))
        results[np_] = {'u_hat': u_hat, 'err_max': err_max, 'err_rmse': err_rmse}
        print(f"  {np_} puntos: max err = {err_max:.4e},  RMSE = {err_rmse:.4e}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ms = t_ref * 1000

    axes[0].plot(ms, u_ref, 'k-', lw=2, label='True $u$')
    colors = ['b', 'r', 'g']
    styles = [':', '--', '-.']
    for (np_, res), c, s in zip(results.items(), colors, styles):
        axes[0].plot(ms, res['u_hat'], linestyle=s, color=c, lw=1,
                     label=f'{np_}pt (RMSE={res["err_rmse"]:.2e})')
    axes[0].set_ylabel('$u$ [V]')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Inverse HFNN: $\\hat{{u}}$ from RK4 $\\omega$, $i$ (T={T:.0e})')

    for (np_, res), c, s in zip(results.items(), colors, styles):
        axes[1].plot(ms, res['u_hat'] - u_ref, linestyle='-', color=c, lw=0.5,
                     alpha=0.8, label=f'{np_}pt error')
    axes[1].set_ylabel('$\\hat{u} - u$ [V]')
    axes[1].set_xlabel('Time [ms]')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/rodo/10Paper/results/test_inverse_hfnn.png', dpi=300, bbox_inches='tight')
    print(f"  Saved to results/test_inverse_hfnn.png")
    plt.close()

    return results


if __name__ == '__main__':
    print("=== Inverse HFNN vs RK4 (sin ruido) ===\n")

    for T in [1e-4, 5e-5, 2e-5]:
        print(f"\nT = {T:.0e}:")
        run_test(T=T, t_final=0.2)
