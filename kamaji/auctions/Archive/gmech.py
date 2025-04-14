import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize


class DecentralizedGMechanism:
    def __init__(self, D, alpha, utility_exprs, dt=0.01):
        """
        Args:
            D (float): Total resource to allocate
            alpha (float): Exponent in marginal price function g(B) = B^alpha / D
            utility_exprs (list of sympy expressions): Utility functions as expressions in `x`
            dt (float): Time step for RK4 integration
        """
        self.D = D
        self.alpha = alpha
        self.dt = dt
        self.N = len(utility_exprs)

        # Compile symbolic derivatives
        x = sp.Symbol('x')
        self.utility_funcs = [sp.lambdify(x, u, modules="numpy") for u in utility_exprs]
        self.utility_derivs = [sp.lambdify(x, sp.diff(u, x), modules="numpy") for u in utility_exprs]

    def compute_optimal_inverse_allocation(self):
        N = self.N

        def total_burden(x_vec):
            return sum(self.utility_funcs[i](x_vec[i]) for i in range(N))

        cons = {'type': 'eq', 'fun': lambda x_vec: np.sum(x_vec) - self.D}
        # bounds = [(1e-8, self.D) for _ in range(N)]
        eps = 1e-6
        bounds = [(eps, self.D - eps) for _ in range(N)]    # avoid numerical issues

        x0 = np.ones(N) * (self.D / N)

        result = minimize(total_burden, x0, constraints=cons, bounds=bounds)

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        return result.x



    def g(self, B):
        return (B ** self.alpha) / self.D

    def compute_derivative(self, bids):
        B = np.sum(bids)
        # allocations = (bids / B) * self.D
        inv_bids = 1 / (bids + 1e-8)  # avoid divide-by-zero
        Z = np.sum(inv_bids)
        allocations = (inv_bids / Z) * self.D

        g_val = self.g(B)
        derivs = np.zeros_like(bids)
        marg_utils = []

        for i in range(self.N):
            Ui_prime = self.utility_derivs[i](allocations[i])
            marg_utils.append(Ui_prime)
            direction = np.sign(Ui_prime - g_val)
            derivs[i] = bids[i] * direction

        return derivs, allocations, marg_utils, g_val

    def rk4_step(self, bids):
        dt = self.dt
        k1, _, _, _ = self.compute_derivative(bids)
        k2, _, _, _ = self.compute_derivative(bids + 0.5 * dt * k1)
        k3, _, _, _ = self.compute_derivative(bids + 0.5 * dt * k2)
        k4, allocs, marg_utils, g_val = self.compute_derivative(bids + dt * k3)

        new_bids = bids + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_bids = np.maximum(new_bids, 1e-8)

        return new_bids, allocs, marg_utils, g_val

    def run(self, initial_bids, steps=1000):
        bids = np.array(initial_bids, dtype=float)
        bid_history = [bids.copy()]
        alloc_history = []
        marg_util_history = []
        g_history = []

        for _ in range(steps):
            bids, allocs, marg_utils, g_val = self.rk4_step(bids)
            bid_history.append(bids.copy())
            alloc_history.append(allocs)
            marg_util_history.append(marg_utils)
            g_history.append(g_val)

        return {
            "bids": np.array(bid_history),
            "allocs": np.array(alloc_history),
            "marg_utils": np.array(marg_util_history),
            "g_vals": np.array(g_history)
        }
    
    def run_until_convergence(self, initial_bids, max_steps=10000, tol=1e-4):
        bids = np.array(initial_bids, dtype=float)
        bid_history = [bids.copy()]
        alloc_history = []
        marg_util_history = []
        g_history = []

        for step in range(max_steps):
            old_bids = bids.copy()
            bids, allocs, marg_utils, g_val = self.rk4_step(bids)

            bid_history.append(bids.copy())
            alloc_history.append(allocs)
            marg_util_history.append(marg_utils)
            g_history.append(g_val)

            # Convergence: relative change in bids
            if np.linalg.norm(bids - old_bids) < tol:
                print(f"✅ Converged after {step+1} steps")
                break
        else:
            print(f"⚠️ Max steps ({max_steps}) reached without full convergence")

        return {
            "bids": np.array(bid_history),
            "allocs": np.array(alloc_history),
            "marg_utils": np.array(marg_util_history),
            "g_vals": np.array(g_history)
        }


    def plot_convergence(self, results):
        bids = results["bids"]
        steps = bids.shape[0]

        plt.figure(figsize=(10, 6))
        for i in range(self.N):
            plt.plot(range(steps), bids[:, i], label=f"Agent {i+1}")
        plt.title("Bid Trajectories (RK4)")
        plt.xlabel("Iteration")
        plt.ylabel("Bid Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_marginals(self, results):
        marg_utils = results["marg_utils"]
        g_vals = results["g_vals"]

        plt.figure(figsize=(10, 6))
        for i in range(self.N):
            plt.plot(marg_utils[:, i], label=f"U'{i+1}(x)")
        plt.plot(g_vals, label="g(B)", linestyle="--", color="black")
        plt.title("Marginal Valuations vs. Marginal Price")
        plt.xlabel("Iteration")
        plt.ylabel("Marginal Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # import sympy as sp
    # x = sp.Symbol('x')
    # utility_exprs = [
    #     sp.log(x),                     # Agent 1: U(x) = log(x)
    #     -1 / x,                        # Agent 2: U(x) = -1/x
    #     sp.log(x + 1),                 # Agent 3: U(x) = log(x + 1)
    #     sp.sqrt(x + 1) - 1             # Agent 4: U(x) = sqrt(x + 1) - 1
    # ]
    # t_final = 10
    # dt = 0.01
    # steps = int(t_final / dt)
    # auction = DecentralizedGMechanism(D=3, alpha=2.0, utility_exprs=utility_exprs, dt=dt)
    # initial_bids = [0.1, 0.1, 0.1, 0.1]
    # results = auction.run(initial_bids, steps=steps)
    # auction.plot_convergence(results)
    # auction.plot_marginals(results)

    import sympy as sp
    x = sp.Symbol('x')
    utility_exprs = [
        2*(x+1)**2,
        3*(x+1)**2,
        (x+1)**2
    ]
    t_final = 10
    dt = 0.001
    steps = int(t_final / dt)
    auction = DecentralizedGMechanism(D=10, alpha=2.0, utility_exprs=utility_exprs, dt=dt)
    initial_bids = [0.1, 0.1, 0.1]
    results = auction.run(initial_bids, steps=steps)
    results2 = auction.run_until_convergence(initial_bids)
    auction.plot_convergence(results)
    # auction.plot_marginals(results)
    final_allocations = results["allocs"][-1]
    total = np.sum(final_allocations)
    final_allocations2 = results2["allocs"][-1]
    total2 = np.sum(final_allocations2)

    optimal_allocations = auction.compute_optimal_inverse_allocation()

    # print(f"Final Allocations: ({total:.4f})")
    # for i, alloc in enumerate(final_allocations):
    #     print(f"  Agent {i+1}: {alloc:.4f} units ({alloc:.2f})")

    print(f"Final Allocations: ({total2:.4f})")
    for i, alloc in enumerate(final_allocations2):
        print(f"  Agent {i+1}: {alloc:.4f} units ({alloc:.2f})")

    print("\n Optimal vs. Auction Final Allocations:")
    for i, (opt, approx) in enumerate(zip(optimal_allocations, final_allocations)):
        error = 100 * abs(opt - approx) / opt
        print(f"Agent {i+1}: Opt = {opt:.4f}, Auction = {approx:.4f}, Error = {error:.2f}%")

