import numpy as np
from numba import njit
from typing import Tuple, List, Callable
from time import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# @njit(cache=True)
def euler_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    """Euler's method for solving ODEs.

    Args:
        func (Callable[[float, np.ndarray], np.ndarray]): The ODE function to compute the derivatives.
        t (float): The current time.
        y (np.ndarray): The current state (dependent variables).
        u (np.ndarray): The control input array.
        h (float): The time step for integration.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the new time and the new state after the Euler step.
    """
    y_new = y + h * func(t, y, u)
    t_new = t + h
    return t_new, y_new


# @njit(cache=True)
def rk2_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    """Second-order Runge-Kutta (RK2) step method for solving ODEs.

    Args:
        func (Callable[[float, np.ndarray], np.ndarray]): The ODE function to compute the derivatives.
        t (float): The current time.
        y (np.ndarray): The current state (dependent variables).
        u (np.ndarray): The control input array.
        h (float): The time step for integration.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the new time and the new state after the RK2 step.
    """
    k1 = func(t, y, u)
    k2 = func(t + h / 2, y + h / 2 * k1, u)
    y_new = y + h * k2
    t_new = t + h
    return t_new, y_new


# @njit(cache=True)
def rk4_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    """Runge-Kutta 4th-order (RK4) step method for solving ODEs.

    Args:
        func (Callable[[float, np.ndarray], np.ndarray]): The ODE function to compute the derivatives.
        t (float): The current time.
        y (np.ndarray): The current state (dependent variables).
        u (np.ndarray): The control input array.
        h (float): The time step for integration.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the new time and the new state after the RK4 step.
    """
    k1 = func(t, y, u)
    k2 = func(t + h / 2, y + h / 2 * k1, u)
    k3 = func(t + h / 2, y + h / 2 * k2, u)
    k4 = func(t + h, y + h * k3, u)
    y_new = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t_new = t + h
    return t_new, y_new


# @njit(cache=True)
def rk45_step(func, t: float, y: np.ndarray, u: np.ndarray, h: float, tol: float = 1e-4, max_step: float = 0.01) -> \
Tuple[float, np.ndarray, float]:
    """Runge-Kutta-Fehlberg (RK45) adaptive step method for solving ODEs with improved accuracy.

    Args:
        func (Callable[[float, np.ndarray], np.ndarray]): The ODE function to compute the derivatives.
        t (float): The current time.
        y (np.ndarray): The current state (dependent variables).
        u (np.ndarray): The control input array.
        h (float): The time step for integration.
        tol (float, optional): The tolerance for adaptive step size control. Defaults to 1e-6.
        max_step (float, optional): The maximum allowed time step. Defaults to None.

    Returns:
        Tuple[float, np.ndarray, float]: A tuple containing the new time, the new state after the RK45 step, and the updated time step size.
    """
    # Coefficients for Dormand-Prince RK45 method
    b1, b2, b3, b4, b5, b6 = 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    b1_star, b2_star, b3_star, b4_star, b5_star, b6_star = 5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100
    c2, c3, c4, c5, c6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656

    # Runge-Kutta steps
    k1 = func(t, y, u)
    k2 = func(t + c2 * h, y + h * a21 * k1, u)
    k3 = func(t + c3 * h, y + h * (a31 * k1 + a32 * k2), u)
    k4 = func(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3), u)
    k5 = func(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), u)
    k6 = func(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), u)

    # Fourth-order and fifth-order approximations
    y_rk4 = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    y_rk45 = y + h * (b1_star * k1 + b3_star * k3 + b4_star * k4 + b5_star * k5 + b6_star * k6)

    # Improved error estimate using relative and absolute tolerances
    abs_err = np.abs(y_rk45 - y_rk4)
    rel_err = np.maximum(np.abs(y_rk45), np.abs(y_rk4))
    err = np.linalg.norm(abs_err / np.maximum(rel_err, tol)) / np.sqrt(len(y))  # Root mean square error

    # Control step size using a more refined strategy
    if err > tol:
        # Reduce step size (more conservative factor)
        h = h * max(0.1, 0.8 * (tol / err) ** 0.2)
    else:
        # Accept the RK45 result
        t += h
        y = y_rk45.copy()
        if err < tol / 2:
            # Increase step size, but within bounds
            h = min(h * 1.5, max_step if max_step is not None else h * 1.5)

    return t, y, h


def solve_ode(func, method: str, t0: float, y0: np.ndarray, h: float, t_end: float) -> List[Tuple[float, np.ndarray]]:
    """Solves an ODE using the specified method.

    Args:
        func (Callable[[float, np.ndarray], np.ndarray]): The ODE function to compute the derivatives.
        method (str): The ODE solving method to use ('euler', 'rk2', 'rk4', or 'rk45').
        t0 (float): The initial time.
        y0 (np.ndarray): The initial state.
        h (float): The time step for integration.
        t_end (float): The end time for the simulation.

    Returns:
        list[Tuple[float, np.ndarray]]: A list of tuples, each containing the time and state at each step.
    """
    t, y = t0, y0
    solution = [(t, y.copy())]  # Start with initial condition

    while t < t_end:
        if method == 'euler':
            t, y = euler_step(func, t, y, h)
        elif method == 'rk2':
            t, y = rk2_step(func, t, y, h)
        elif method == 'rk4':
            t, y = rk4_step(func, t, y, h)
        elif method == 'rk45':
            t, y, h = rk45_step(func, t, y, h)
        solution.append((t, y.copy()))

    return solution


if __name__ == "__main__":
    # System of ODE: Simple Harmonic Motion
    @njit(cache=True)
    def simple_harmonic_motion(t: float, y: np.ndarray) -> np.ndarray:
        """Defines the system of ODEs for simple harmonic motion."""
        return np.array([y[1], -y[0]])


    # Define a sample ODE function for testing
    @njit
    def ode_function(t, y):
        return -y  # Example ODE


    # Define the initial conditions and time span
    t0 = 0.0
    y0 = np.array([1.0, 0.0])  # Initial position = 1, initial velocity = 0
    h = 0.01  # Time step for fixed-step methods
    t_end = 10.0  # Solve from t=0 to t=100

    for _ in range(10):
        # Call each method to trigger compilation
        # euler_step(simple_harmonic_motion, t_initial, y_initial, h)
        # rk2_step(simple_harmonic_motion, t_initial, y_initial, h)
        # rk4_step(simple_harmonic_motion, t_initial, y_initial, h)
        solve_ode(simple_harmonic_motion, "euler", t0, y0, h, t_end)
        solve_ode(simple_harmonic_motion, "rk2", t0, y0, h, t_end)
        solve_ode(simple_harmonic_motion, "rk4", t0, y0, h, t_end)
        # rk45_step(simple_harmonic_motion, t_initial, y_initial, h)


    # Exact solution for simple harmonic motion (SHM)
    @njit(cache=True)
    def exact_solution(t: float) -> np.ndarray:
        """Exact solution for SHM: y(t) = cos(t), v(t) = -sin(t)."""
        return np.array([np.cos(t), -np.sin(t)])


    # Benchmark function with accuracy measurement
    def benchmark_with_accuracy(method_name: str, method_func: Callable, scipy=False) -> None:
        if scipy:
            # Solve the ODE using SciPy's RK45
            start_time = time()
            sol = solve_ivp(simple_harmonic_motion, [t0, t_end], y0, method='RK45', max_step=h, rtol=1e-6)
            elapsed_time = time() - start_time

            times = sol.t
            numerical_solutions = sol.y.T
        else:
            # Solve the ODE numerically using the provided method
            start_time = time()
            solution = solve_ode(simple_harmonic_motion, method_name, t0, y0, h, t_end)
            elapsed_time = time() - start_time

            # Calculate the errors by comparing with the exact solution
            times, numerical_solutions = zip(*solution)
            times = np.array(times)
            numerical_solutions = np.array(numerical_solutions)

        exact_solutions = np.array([exact_solution(t) for t in times])

        # Calculate the absolute error
        errors = np.abs(numerical_solutions - exact_solutions)
        max_error = np.max(errors)
        mean_error = np.mean(errors)

        print(f"{method_name.upper()} method:")
        print(f"  Time taken: {elapsed_time:.6f} seconds")
        print(f"  Max error: {max_error:.6e}")
        print(f"  Mean error: {mean_error:.6e}\n")

        # Plot the error over time
        # plt.figure(figsize=(10, 6))
        # plt.plot(times, errors[:, 0], label=f'Error in position (method: {method_name})')
        # plt.plot(times, errors[:, 1], label=f'Error in velocity (method: {method_name})')
        # plt.xlabel('Time')
        # plt.ylabel('Error')
        # plt.title(f'Error over time for {method_name.upper()} method')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


    # Define the initial conditions and time span
    t0 = 0.0
    y0 = np.array([1.0, 0.0])  # Initial position = 1, initial velocity = 0
    h = 0.01  # Time step for fixed-step methods
    t_end = 10000.0  # Solve from t=0 to t=100

    # Benchmark each method

    benchmark_with_accuracy('euler', solve_ode)
    benchmark_with_accuracy('rk2', solve_ode)
    benchmark_with_accuracy('rk4', solve_ode)

    # benchmark_with_accuracy('rk45', solve_ode)
    # benchmark_with_accuracy('scipy_rk45', solve_ode, scipy=True)
