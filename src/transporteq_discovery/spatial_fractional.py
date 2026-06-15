"""Space-fractional Taylor linearization helpers.

This module is intentionally standalone.  The main discovery pipeline imports it
only when the optional spatial-fractional correction is enabled.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, roots_jacobi

EULER_GAMMA = 0.5772156649015329


def _fourier_wavenumbers(n_points: int, dx: float) -> np.ndarray:
    if n_points <= 1:
        raise ValueError("at least two spatial points are required")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    return 2.0 * np.pi * np.fft.fftfreq(n_points, d=dx)


def _ik_power(kappa: np.ndarray, order: float) -> np.ndarray:
    ik = 1j * kappa
    with np.errstate(divide="ignore", invalid="ignore"):
        multiplier = np.where(kappa != 0.0, ik**order, 0.0 + 0.0j)
    return multiplier


def _ik_log(kappa: np.ndarray) -> np.ndarray:
    ik = 1j * kappa
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ik = np.where(kappa != 0.0, np.log(ik), 0.0 + 0.0j)
    return log_ik


def fourier_fractional_derivative(
    u_data: np.ndarray,
    dx: float,
    beta: float,
    *,
    spatial_axis: int = -1,
) -> np.ndarray:
    """Compute the periodic Fourier fractional derivative ``D_x^beta u``.

    This matches the FFT data generator, where the spatial fractional operator
    is the Fourier multiplier ``(i*kappa)^beta``.  The zero wave-number mode is
    set to zero.
    """

    values = np.asarray(u_data, dtype=float)
    moved = np.moveaxis(values, spatial_axis, -1)
    kappa = _fourier_wavenumbers(moved.shape[-1], dx)
    multiplier = _ik_power(kappa, float(beta))
    transformed = np.fft.fft(moved, axis=-1)
    result = np.fft.ifft(transformed * multiplier.reshape((1,) * (moved.ndim - 1) + (-1,)), axis=-1)
    return np.moveaxis(result.real, -1, spatial_axis)


def fourier_correction_column(
    u_data: np.ndarray,
    dx: float,
    beta_0: float,
    *,
    spatial_axis: int = -1,
) -> np.ndarray:
    """Compute ``partial_beta D_x^beta u`` at ``beta_0`` by FFT/IFFT."""

    values = np.asarray(u_data, dtype=float)
    moved = np.moveaxis(values, spatial_axis, -1)
    kappa = _fourier_wavenumbers(moved.shape[-1], dx)
    multiplier = _ik_power(kappa, float(beta_0)) * _ik_log(kappa)
    transformed = np.fft.fft(moved, axis=-1)
    result = np.fft.ifft(transformed * multiplier.reshape((1,) * (moved.ndim - 1) + (-1,)), axis=-1)
    return np.moveaxis(result.real, -1, spatial_axis)


def build_spatial_fractional_candidates(
    u_data: np.ndarray,
    dx: float,
    beta_0: float,
    *,
    spatial_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build Fourier main and Taylor correction columns for one ``beta_0``."""

    beta0_float = float(beta_0)
    is_integer = bool(np.isclose(beta0_float, round(beta0_float)))
    main = fourier_fractional_derivative(u_data, dx, beta0_float, spatial_axis=spatial_axis)
    correction = fourier_correction_column(u_data, dx, beta0_float, spatial_axis=spatial_axis)
    return main, correction, is_integer

# Milovanovic-style Gauss-log nodes and weights for w(s)=log(1/s) on [0, 1].
GAUSS_LOG_NODES_10 = np.array(
    [
        0.00904250467020043837,
        0.0539705582010104254,
        0.135310294818498794,
        0.24705015887763554,
        0.380209874571341644,
        0.523789641135512074,
        0.66577288473727414,
        0.794188723616433001,
        0.898160139184751127,
        0.968847672108813507,
    ],
    dtype=float,
)
GAUSS_LOG_WEIGHTS_10 = np.array(
    [
        0.120953802941505928,
        0.186362164253248502,
        0.195660328432840369,
        0.173577449160700503,
        0.135696451682628699,
        0.0936476169396245617,
        0.0557884141962017369,
        0.0271602272374394209,
        0.00951535455917018949,
        0.00163819059663946709,
    ],
    dtype=float,
)

# The tabulated 5-point rule below is kept for low-cost discovery experiments.
GAUSS_LOG_NODES_5 = np.array(
    [
        0.0291344721519654938,
        0.173977213320890139,
        0.411702520284955575,
        0.677314174582931305,
        0.894771361031075019,
    ],
    dtype=float,
)
GAUSS_LOG_WEIGHTS_5 = np.array(
    [
        0.297893471782857966,
        0.349776226513281607,
        0.234488290044090691,
        0.0989304595165953482,
        0.0189115521431739778,
    ],
    dtype=float,
)


def gauss_log_quadrature_golub_welsch(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate an n-point Gauss rule for int_0^1 f(t) log(1/t) dt.

    The moments are m_k = int_0^1 t^k log(1/t) dt = 1/(k+1)^2.  A simple
    Stieltjes procedure on monic polynomials builds the Jacobi matrix used by
    Golub-Welsch.  This is primarily a backup for orders other than the
    hardcoded tables.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    moments = np.array([1.0 / (k + 1) ** 2 for k in range(2 * n + 1)], dtype=float)

    def inner(p: np.ndarray, q: np.ndarray) -> float:
        total = 0.0
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                total += float(pi) * float(qj) * moments[i + j]
        return total

    polys: list[np.ndarray] = [np.array([1.0])]
    norms = [inner(polys[0], polys[0])]
    alpha = np.zeros(n, dtype=float)
    beta = np.zeros(n, dtype=float)
    for k in range(n):
        p = polys[k]
        xp = np.concatenate(([0.0], p))
        alpha[k] = inner(xp, p) / norms[k]
        if k == n - 1:
            break
        beta[k] = 0.0 if k == 0 else norms[k] / norms[k - 1]
        next_poly = xp - alpha[k] * np.pad(p, (0, 1))
        if k > 0:
            next_poly -= beta[k] * np.pad(polys[k - 1], (0, 2))
        polys.append(next_poly)
        norms.append(inner(next_poly, next_poly))

    jacobi = np.diag(alpha)
    for k in range(1, n):
        offdiag = np.sqrt(norms[k] / norms[k - 1])
        jacobi[k - 1, k] = offdiag
        jacobi[k, k - 1] = offdiag
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = moments[0] * vectors[0, :] ** 2
    order = np.argsort(nodes)
    return nodes[order], weights[order]


def _gauss_log_rule(n_quad: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "golub_welsch":
        return gauss_log_quadrature_golub_welsch(n_quad)
    if method != "hardcoded":
        raise ValueError("method must be 'hardcoded' or 'golub_welsch'")
    if n_quad == 10:
        return GAUSS_LOG_NODES_10, GAUSS_LOG_WEIGHTS_10
    if n_quad == 5:
        return GAUSS_LOG_NODES_5, GAUSS_LOG_WEIGHTS_5
    return gauss_log_quadrature_golub_welsch(n_quad)


def log_kernel_convolution(
    u_values: np.ndarray,
    x_grid: np.ndarray,
    n_quad: int = 10,
    method: str = "hardcoded",
) -> np.ndarray:
    """Compute I[u](x)=int_0^x [-ln(x-tau)] u(tau) dtau.

    The spatial axis is the last axis of ``u_values``.  One-dimensional inputs
    are returned as one-dimensional arrays.
    """

    values = np.asarray(u_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1:
        raise ValueError("x_grid must be one-dimensional")
    if values.shape[-1] != x_grid.size:
        raise ValueError("last axis of u_values must match x_grid")
    if np.any(np.diff(x_grid) <= 0):
        raise ValueError("x_grid must be strictly increasing")

    log_nodes, log_weights = _gauss_log_rule(n_quad, method)
    leg_nodes_raw, leg_weights_raw = np.polynomial.legendre.leggauss(n_quad)
    leg_nodes = 0.5 * (leg_nodes_raw + 1.0)
    leg_weights = 0.5 * leg_weights_raw

    original_shape = values.shape
    flat_values = values.reshape((-1, x_grid.size))
    result = np.zeros_like(flat_values, dtype=float)

    for row_index, row in enumerate(flat_values):
        for x_index, x_value in enumerate(x_grid):
            if x_value <= 0.0:
                result[row_index, x_index] = 0.0
                continue
            log_eval = x_value * (1.0 - log_nodes)
            smooth_eval = x_value * (1.0 - leg_nodes)
            item_a = np.dot(log_weights, np.interp(log_eval, x_grid, row))
            item_b = np.dot(leg_weights, np.interp(smooth_eval, x_grid, row))
            result[row_index, x_index] = x_value * item_a + x_value * np.log(1.0 / x_value) * item_b
    return result.reshape(original_shape)


def homogenize_left_boundary(
    u_values: np.ndarray,
    x_grid: np.ndarray,
    *,
    mode: str = "left_constant_subtract",
    boundary_value: float | np.ndarray | None = None,
    boundary_slope: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return the spatial field used by the left fractional operator.

    The raw RL operator treats a nonzero constant left boundary as part of the
    fractional derivative.  For the FADE benchmarks the inlet value at ``x=0``
    is a prescribed constant, so the default mode subtracts that value before
    applying the fractional spatial operator.  ``left_caputo`` also subtracts
    the linear boundary jet and corresponds to the usual Caputo correction for
    ``1 < beta < 2``.
    """

    values = np.asarray(u_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.shape[-1] != x_grid.size:
        raise ValueError("last axis of u_values must match x_grid")
    if x_grid.ndim != 1:
        raise ValueError("x_grid must be one-dimensional")

    mode_key = mode.lower()
    if mode_key == "raw":
        return values.copy()
    if mode_key not in {"left_constant_subtract", "left_caputo"}:
        raise ValueError("mode must be 'raw', 'left_constant_subtract', or 'left_caputo'")

    if boundary_value is None:
        left_value = values[..., :1]
    else:
        left_value = np.asarray(boundary_value, dtype=float)
        if left_value.ndim == 0:
            left_value = np.full(values.shape[:-1] + (1,), float(left_value))
        elif left_value.shape == values.shape[:-1]:
            left_value = left_value[..., np.newaxis]
        if left_value.shape != values.shape[:-1] + (1,):
            raise ValueError("boundary_value must be scalar or match u_values without the spatial axis")

    homogenized = values - left_value
    if mode_key == "left_constant_subtract":
        return homogenized

    if boundary_slope is None:
        slope = np.gradient(values, x_grid, axis=-1, edge_order=2)[..., :1]
    else:
        slope = np.asarray(boundary_slope, dtype=float)
        if slope.ndim == 0:
            slope = np.full(values.shape[:-1] + (1,), float(slope))
        elif slope.shape == values.shape[:-1]:
            slope = slope[..., np.newaxis]
        if slope.shape != values.shape[:-1] + (1,):
            raise ValueError("boundary_slope must be scalar or match u_values without the spatial axis")
    return homogenized - slope * (x_grid - x_grid[0])


def spatial_correction_column(
    u_data: np.ndarray,
    x_grid: np.ndarray,
    dx: float,
    beta_0: float = 2.0,
    n_quad: int = 10,
    u_xx_values: np.ndarray | None = None,
    method: str = "hardcoded",
    correction_method: str = "gj_richardson",
    derivative_step: float = 2.0e-2,
) -> np.ndarray:
    """Compute the spatial Taylor correction column K[u].

    ``correction_method`` selects the numerical approximation for
    partial_beta D_x^beta u at beta_0.  The log-kernel formula is retained as
    an experimental beta0=2 path; the default uses G-J finite differences with
    Richardson extrapolation.
    """

    values = np.asarray(u_data, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.shape[-1] != x_grid.size:
        raise ValueError("last axis of u_data must match x_grid")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    correction_key = correction_method.lower()
    if correction_key in {"gj_finite_diff", "gj_richardson", "trapz_reference", "gl_pycaputo"}:
        return spatial_order_derivative_beta(
            values,
            x_grid,
            beta_0=float(beta_0),
            n_quad=n_quad,
            method=correction_key,
            derivative_step=derivative_step,
            u_xx_values=u_xx_values,
        )[1]
    if correction_key != "log_kernel":
        raise ValueError(
            "correction_method must be 'gj_finite_diff', 'gj_richardson', "
            "'gl_pycaputo', 'log_kernel', or 'trapz_reference'"
        )
    if not np.isclose(beta_0, 2.0):
        raise ValueError("log_kernel correction is only available for beta_0=2.0")

    convolution = log_kernel_convolution(values, x_grid, n_quad=n_quad, method=method)
    if x_grid.size < 3:
        raise ValueError("at least three spatial points are required for second derivatives")
    first = np.gradient(convolution, x_grid, axis=-1, edge_order=2)
    second = np.gradient(first, x_grid, axis=-1, edge_order=2)
    if u_xx_values is None:
        first_u = np.gradient(values, x_grid, axis=-1, edge_order=2)
        u_xx = np.gradient(first_u, x_grid, axis=-1, edge_order=2)
    else:
        u_xx = np.asarray(u_xx_values, dtype=float)
        if u_xx.shape != values.shape:
            raise ValueError("u_xx_values must have the same shape as u_data")
    correction = second + EULER_GAMMA * u_xx
    if not np.all(np.isfinite(correction)):
        raise FloatingPointError("spatial correction generated NaN or Inf values")
    return correction


def rl_space_derivative(
    u_values: np.ndarray,
    x_grid: np.ndarray,
    beta: float,
    n_quad: int = 10,
) -> np.ndarray:
    """Approximate the left RL derivative ``D_x^beta u`` for ``1<beta<2``.

    The implementation follows the Jacobi-Gauss form used by the reference
    DL-FDE code, then applies two finite-difference derivatives in ``x``.
    The spatial axis is the last axis of ``u_values``.
    """

    if not 1.0 < beta < 2.0:
        raise ValueError("rl_space_derivative currently supports 1 < beta < 2")
    values = np.asarray(u_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.shape[-1] != x_grid.size:
        raise ValueError("last axis of u_values must match x_grid")
    if x_grid.ndim != 1 or np.any(np.diff(x_grid) <= 0):
        raise ValueError("x_grid must be one-dimensional and strictly increasing")

    jacobi_nodes, jacobi_weights = roots_jacobi(n_quad, 0.0, 1.0 - beta)
    flat_values = values.reshape((-1, x_grid.size))
    fractional_integral = np.zeros_like(flat_values, dtype=float)
    prefactor = 1.0 / gamma(2.0 - beta)

    for row_index, row in enumerate(flat_values):
        for x_index, x_value in enumerate(x_grid):
            if x_value <= 0.0:
                fractional_integral[row_index, x_index] = 0.0
                continue
            sample_x = x_value - x_value * 0.5 * (jacobi_nodes + 1.0)
            sampled = np.interp(sample_x, x_grid, row)
            fractional_integral[row_index, x_index] = (
                np.dot(sampled, jacobi_weights)
                * prefactor
                * (x_value * 0.5) ** (2.0 - beta)
            )

    integral_grid = fractional_integral.reshape(values.shape)
    first = np.gradient(integral_grid, x_grid, axis=-1, edge_order=2)
    second = np.gradient(first, x_grid, axis=-1, edge_order=2)
    if not np.all(np.isfinite(second)):
        raise FloatingPointError("RL spatial derivative generated NaN or Inf values")
    return second


def rl_space_derivative_trapz_reference(
    u_values: np.ndarray,
    x_grid: np.ndarray,
    beta: float,
    n_points: int = 800,
) -> np.ndarray:
    """Slow reference RL derivative using adaptive quadrature in the transformed variable."""

    from scipy.integrate import quad

    if not 1.0 < beta < 2.0:
        raise ValueError("rl_space_derivative_trapz_reference currently supports 1 < beta < 2")
    values = np.asarray(u_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.shape[-1] != x_grid.size:
        raise ValueError("last axis of u_values must match x_grid")
    del n_points
    flat_values = values.reshape((-1, x_grid.size))
    fractional_integral = np.zeros_like(flat_values, dtype=float)
    prefactor = 1.0 / gamma(2.0 - beta)
    for row_index, row in enumerate(flat_values):
        for x_index, x_value in enumerate(x_grid):
            if x_value <= 0.0:
                fractional_integral[row_index, x_index] = 0.0
                continue
            def integrand(s_value: float) -> float:
                return s_value ** (1.0 - beta) * float(
                    np.interp(x_value * (1.0 - s_value), x_grid, row)
                )

            integral = quad(integrand, 0.0, 1.0, points=[0.0], limit=100)[0]
            fractional_integral[row_index, x_index] = prefactor * x_value ** (2.0 - beta) * integral
    integral_grid = fractional_integral.reshape(values.shape)
    first = np.gradient(integral_grid, x_grid, axis=-1, edge_order=2)
    second = np.gradient(first, x_grid, axis=-1, edge_order=2)
    if not np.all(np.isfinite(second)):
        raise FloatingPointError("trapz RL spatial derivative generated NaN or Inf values")
    return second


def gl_space_derivative_pycaputo(
    u_values: np.ndarray,
    x_grid: np.ndarray,
    beta: float,
    *,
    shift: float = 0.0,
) -> np.ndarray:
    """Approximate the left RL derivative with pycaputo G-L stencils."""

    from pycaputo.differentiation.grunwald_letnikov import ShiftedGrunwaldLetnikov, diff
    from pycaputo.grid import make_uniform_points

    if not 1.0 < beta < 2.0:
        raise ValueError("gl_space_derivative_pycaputo currently supports 1 < beta < 2")
    values = np.asarray(u_values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if values.shape[-1] != x_grid.size:
        raise ValueError("last axis of u_values must match x_grid")
    if x_grid.ndim != 1 or np.any(np.diff(x_grid) <= 0):
        raise ValueError("x_grid must be one-dimensional and strictly increasing")

    points = make_uniform_points(x_grid.size, a=float(x_grid[0]), b=float(x_grid[-1]))
    method = ShiftedGrunwaldLetnikov(alpha=float(beta), shift=float(shift))
    flat_values = values.reshape((-1, x_grid.size))
    result = np.zeros_like(flat_values, dtype=float)
    for row_index, row in enumerate(flat_values):
        def row_function(query_x: np.ndarray | float) -> np.ndarray:
            query = np.asarray(query_x, dtype=float)
            return np.interp(query, x_grid, row)

        derivative = np.asarray(diff(method, row_function, points), dtype=float)
        if derivative.size != x_grid.size:
            raise ValueError("pycaputo G-L derivative returned an unexpected shape")
        if derivative.size >= 2 and not np.isfinite(derivative[0]):
            derivative[0] = derivative[1]
        result[row_index, :] = derivative
    reshaped = result.reshape(values.shape)
    if not np.all(np.isfinite(reshaped)):
        raise FloatingPointError("pycaputo G-L spatial derivative generated NaN or Inf values")
    return reshaped


def spatial_order_derivative_beta(
    u_values: np.ndarray,
    x_grid: np.ndarray,
    beta_0: float,
    n_quad: int = 10,
    method: str = "gj_richardson",
    derivative_step: float = 2.0e-2,
    u_xx_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``D_x^beta0 u`` and ``partial_beta D_x^beta u|beta0``.

    For beta0=2 a left finite difference is used to avoid evaluating
    undefined beta>2 orders.  For 1<beta0<2 the default method applies a
    centered finite difference plus Richardson extrapolation.
    """

    method_key = method.lower()
    if method_key not in {"gj_finite_diff", "gj_richardson", "trapz_reference", "gl_pycaputo"}:
        raise ValueError("method must be 'gj_finite_diff', 'gj_richardson', 'trapz_reference', or 'gl_pycaputo'")
    if not 1.0 < beta_0 <= 2.0:
        raise ValueError("spatial_order_derivative_beta supports only 1 < beta_0 <= 2")

    def derivative_at(order: float) -> np.ndarray:
        if np.isclose(order, 2.0):
            if u_xx_values is not None:
                return np.asarray(u_xx_values, dtype=float)
            first_u = np.gradient(u_values, x_grid, axis=-1, edge_order=2)
            return np.gradient(first_u, x_grid, axis=-1, edge_order=2)
        if method_key == "trapz_reference":
            return rl_space_derivative_trapz_reference(u_values, x_grid, order)
        if method_key == "gl_pycaputo":
            return gl_space_derivative_pycaputo(u_values, x_grid, order)
        return rl_space_derivative(u_values, x_grid, order, n_quad=n_quad)

    max_step = 0.45 * (beta_0 - 1.0)
    if beta_0 < 2.0:
        max_step = min(max_step, 0.45 * (2.0 - beta_0))
    step = min(float(derivative_step), max_step)
    if step <= 0.0:
        raise ValueError("derivative_step is too small for the requested beta_0")
    main = derivative_at(beta_0)

    if np.isclose(beta_0, 2.0):
        d1 = (main - derivative_at(beta_0 - step)) / step
        if method_key == "gj_richardson":
            half = 0.5 * step
            d2 = (main - derivative_at(beta_0 - half)) / half
            correction = 2.0 * d2 - d1
        else:
            correction = d1
    else:
        plus = derivative_at(beta_0 + step)
        minus = derivative_at(beta_0 - step)
        d1 = (plus - minus) / (2.0 * step)
        if method_key == "gj_richardson":
            half = 0.5 * step
            plus_half = derivative_at(beta_0 + half)
            minus_half = derivative_at(beta_0 - half)
            d2 = (plus_half - minus_half) / (2.0 * half)
            correction = (4.0 * d2 - d1) / 3.0
        else:
            correction = d1
    if not np.all(np.isfinite(main)) or not np.all(np.isfinite(correction)):
        raise FloatingPointError("spatial beta derivative generated NaN or Inf values")
    return main, correction
