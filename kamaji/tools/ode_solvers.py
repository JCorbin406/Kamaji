import numpy as np
from typing import Tuple, List, Callable


def euler_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    """Euler's method for solving ODEs.

    Args:
        func: The ODE function f(t, y, u) -> dy/dt.
        t: The current time.
        y: The current state vector.
        u: The control input array.
        h: The time step.

    Returns:
        Tuple of (new_time, new_state).
    """
    y_new = y + h * func(t, y, u)
    t_new = t + h
    return t_new, y_new


def rk2_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    """Second-order Runge-Kutta (midpoint method) step.

    Args:
        func: The ODE function f(t, y, u) -> dy/dt.
        t: The current time.
        y: The current state vector.
        u: The control input array.
        h: The time step.

    Returns:
        Tuple of (new_time, new_state).
    """
    k1 = func(t, y, u)
    k2 = func(t + h / 2, y + h / 2 * k1, u)
    y_new = y + h * k2
    t_new = t + h
    return t_new, y_new


def rk4_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    """Fourth-order Runge-Kutta step.

    Args:
        func: The ODE function f(t, y, u) -> dy/dt.
        t: The current time.
        y: The current state vector.
        u: The control input array.
        h: The time step.

    Returns:
        Tuple of (new_time, new_state).
    """
    k1 = func(t, y, u)
    k2 = func(t + h / 2, y + h / 2 * k1, u)
    k3 = func(t + h / 2, y + h / 2 * k2, u)
    k4 = func(t + h, y + h * k3, u)
    y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t_new = t + h
    return t_new, y_new


def rk45_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float, tol: float = 1e-4, max_step: float = 0.01) -> \
        Tuple[float, np.ndarray, float]:
    """Dormand-Prince RK45 adaptive step.

    On a rejected step (error > tol), retries with a smaller h until the step
    is accepted, rather than returning stale state.

    Args:
        func: The ODE function f(t, y, u) -> dy/dt.
        t: The current time.
        y: The current state vector.
        u: The control input array.
        h: The initial time step (adapted internally).
        tol: Error tolerance for step acceptance.
        max_step: Maximum allowed step size.

    Returns:
        Tuple of (new_time, new_state, updated_step_size).
    """
    # Dormand-Prince coefficients
    b1, b3, b4, b5, b6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    b1s, b3s, b4s, b5s, b6s = 5179 / 57600, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100
    c2, c3, c4, c5, c6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656

    min_h = h * 1e-6  # safety floor to prevent infinite loops

    while True:
        k1 = func(t, y, u)
        k2 = func(t + c2 * h, y + h * a21 * k1, u)
        k3 = func(t + c3 * h, y + h * (a31 * k1 + a32 * k2), u)
        k4 = func(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3), u)
        k5 = func(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), u)
        k6 = func(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), u)

        y_rk4 = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        y_rk5 = y + h * (b1s * k1 + b3s * k3 + b4s * k4 + b5s * k5 + b6s * k6)

        abs_err = np.abs(y_rk5 - y_rk4)
        scale = np.maximum(np.maximum(np.abs(y_rk5), np.abs(y_rk4)), tol)
        err = np.linalg.norm(abs_err / scale) / np.sqrt(len(y))

        if err <= tol or h <= min_h:
            # Accept the higher-order (5th) result
            t_new = t + h
            y_new = y_rk5.copy()
            # Grow step size if error is well below tolerance
            if err < tol / 2 and err > 0:
                h = min(h * min(5.0, 0.9 * (tol / err) ** 0.2), max_step)
            return t_new, y_new, h
        else:
            # Reject step: shrink h and retry
            h = h * max(0.1, 0.8 * (tol / err) ** 0.2)


def solve_ode(func, method: str, t0: float, y0: np.ndarray, u: np.ndarray,
              h: float, t_end: float) -> List[Tuple[float, np.ndarray]]:
    """Solve an ODE using the specified fixed-step or adaptive method.

    Args:
        func: The ODE function f(t, y, u) -> dy/dt.
        method: One of 'euler', 'rk2', 'rk4', 'rk45'.
        t0: Initial time.
        y0: Initial state vector.
        u: Control input (held constant over the integration).
        h: Time step (initial step for rk45).
        t_end: End time.

    Returns:
        List of (time, state) tuples at each accepted step.
    """
    t, y = t0, y0
    solution = [(t, y.copy())]

    while t < t_end:
        if method == 'euler':
            t, y = euler_step(func, t, y, u, h)
        elif method == 'rk2':
            t, y = rk2_step(func, t, y, u, h)
        elif method == 'rk4':
            t, y = rk4_step(func, t, y, u, h)
        elif method == 'rk45':
            t, y, h = rk45_step(func, t, y, u, h)
        else:
            raise ValueError(f"Unknown ODE method: {method}")
        solution.append((t, y.copy()))

    return solution
