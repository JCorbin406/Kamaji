import numpy as np
from scipy.optimize import minimize


def solve_optimal_allocation_scipy(valuation_funcs, total_supply, x0=None):
    N = len(valuation_funcs)

    def objective(x):
        return sum(v(xi) for v, xi in zip(valuation_funcs, x))

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_supply}
    bounds = [(0, None)] * N

    if x0 is None:
        x0 = np.ones(N) * (total_supply / N)

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[cons])

    if not result.success:
        raise RuntimeError("Optimization failed:", result.message)

    return result.x, result.fun


class InversePreferenceAuction:
    def __init__(self, agent_interfaces, cost_func, supply_limit,
                 rho_bar=1.0, sigma_bar=1.0, epsilon=1e-4, max_iters=1000):
        self.N = len(agent_interfaces)
        self.agents = agent_interfaces  # Each agent must implement get_marginal_disutility(x)
        self.c_func = cost_func
        self.h = supply_limit
        self.rho_bar = rho_bar
        self.sigma_bar = sigma_bar
        self.epsilon = epsilon
        self.max_iters = max_iters

        self.potential_quantity = self.h
        self.potential_quantity_prior = 0.0
        self.allocations = np.zeros(self.N)

        self.bids = [(1.0, 0.0) for _ in range(self.N)]
        for i in range(self.N):
            x_i = self.allocations[i]
            self.bids[i] = (self.agents[i].get_marginal_disutility(x_i), 0.0)

    def cost_derivative(self, y):
        delta = 1e-6
        return (self.c_func(y + delta) - self.c_func(y)) / delta

    def allocate_buyers(self):
        epsilon = 1e-6
        weights = np.array([1.0 / (self.bids[i][0] + epsilon) for i in range(self.N)])
        total_weight = np.sum(weights)
        x = self.potential_quantity * (weights / total_weight)
        self.allocations = x
        return x

    def compute_matched_prices(self):
        allocated = [i for i in range(self.N) if self.allocations[i] > 0]
        if allocated:
            # pb = np.mean([self.bids[i][0] for i in allocated])
            pb = np.min([self.bids[i][0] for i in allocated])
        else:
            pb = 0.0
        ps = self.cost_derivative(self.potential_quantity)
        return pb, ps

    def choose_buyer_to_update(self):
        return np.random.randint(self.N)

    def run(self, verbose=False):
        for k in range(self.max_iters):
            old_bids = list(self.bids)

            self.allocate_buyers()
            pb, ps = self.compute_matched_prices()
            
            self.potential_quantity_prior = self.potential_quantity
            # self.potential_quantity = self.h + (pb - ps) / (self.rho_bar + self.sigma_bar)
            # Q = min(np.sum(self.allocations), self.h - np.sum(self.allocations))
            # Q = np.sum(self.allocations)
            # self.potential_quantity = Q + (pb - ps) / (self.rho_bar + self.sigma_bar)
            self.potential_quantity = self.h

            # Update one buyer's bid using their own marginal disutility function
            i = self.choose_buyer_to_update()
            x_i = max(1e-6, self.allocations[i])
            beta_i = max(1e-6, self.agents[i].get_marginal_disutility(x_i))
            self.bids[i] = (beta_i, 0.0)

            bid_diff = sum(
                abs(b1[0] - b0[0])
                for b0, b1 in zip(old_bids, self.bids)
            )
            epsilon_k = abs(bid_diff) + abs(self.potential_quantity - self.potential_quantity_prior)

            if verbose:
                print(f"Iter {k+1:3d} | ε = {epsilon_k:.5e} | Allocated = {np.sum(self.allocations):.4f} / {self.h} | p_b = {pb:.4f}, p_s = {ps:.4f}")
                # print(f"Iter {k+1:3d} | A1 = {self.allocations[0]:.5e} | A2 = {self.allocations[1]:.5e} | A3 = {self.allocations[2]:.5e} |")
            
            if epsilon_k < self.epsilon:
                if verbose:
                    print(f"Converged in {k+1} iterations.")
                break

        return self.allocations, self.bids, self.potential_quantity


class Agent:
    def __init__(self, marginal_disutility_func):
        self.marginal_disutility_func = marginal_disutility_func

    def get_marginal_disutility(self, x):
        return self.marginal_disutility_func(x)


if __name__ == "__main__":
    v_funcs = [
        lambda x: (x)**2,
        lambda x: 10 * (x)**2,
        lambda x: (x)**2
    ]

    marginal_derivs = [
        lambda x: 2 * (x),
        lambda x: 20 * (x),
        lambda x: 2 * (x)
    ]

    agents = [Agent(m) for m in marginal_derivs]
    c_func = lambda y: y**2

    auction = InversePreferenceAuction(
        agent_interfaces=agents,
        cost_func=c_func,
        supply_limit=10.0,
        rho_bar=10.0,
        sigma_bar=0.0,
        epsilon=1e-6
    )

    auction_allocs, bids, _ = auction.run(verbose=True)
    print("\nAuction Allocations:", auction_allocs)

    for i in range(len(auction_allocs)):
        x = auction_allocs[i]
        beta = bids[i][0]
        print(f"Agent {i}: x = {x:.4f}, beta = {beta:.4f}")

    opt_allocs, opt_val = solve_optimal_allocation_scipy(v_funcs, total_supply=10.0)
    print("\nOptimal Allocations:", opt_allocs)
    diff_norm = np.linalg.norm(auction_allocs - opt_allocs, ord=2)
    print(f"Difference (L2 norm): {diff_norm:.4e}")
