from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt

class FixedResourceAuction:
    def __init__(
        self,
        cost_funcs: List[Callable[[float], float]],
        R: float,           # total resource to allocate
        C: float,           # target (constant) distribution cost (dummy buyer's fixed marginal value)
        h_init: np.ndarray, # initial bid supplies for each agent (seller)
        epsilon: float = 1e-3,
        max_iter: int = 5000,
        eta: float = 0.05   # step size for seller update
    ):
        self.N = len(cost_funcs)  # number of agents (sellers)
        self.cost_funcs = cost_funcs
        self.R = R
        self.C = C  # target equilibrium marginal cost
        self.h = h_init.copy()  # sellers' bid supplies
        self.alpha = np.array([self.cp(i, self.h[i]) for i in range(self.N)])
        # Dummy buyer: fixed bid demand R and fixed bid price = C.
        self.d_dummy = R  
        self.beta_dummy = C  
        # We fix Gamma to equal R (full allocation)
        self.Gamma = R
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.eta = eta
        self.history = []  # record iteration states

    def cp(self, i, y):
        # Finite difference approximation of the i-th seller's marginal cost.
        delta = 1e-6
        return (self.cost_funcs[i](y + delta) - self.cost_funcs[i](y)) / delta

    def allocate_sellers(self):
        """
        Allocate the fixed resource R among sellers.
        If the total seller supply is less than R, then scale allocations
        proportionally so that the total allocation equals R.
        Otherwise, allocate in ascending order of alpha.
        """
        total_supply = np.sum(self.h)
        if total_supply < self.R:
            # Scale up each seller's supply proportionally.
            return self.h * (self.R / total_supply)
        else:
            # Standard allocation: allocate R by giving each seller up to h[i] in ascending order of alpha.
            sorted_indices = np.argsort(self.alpha)
            remaining = self.R
            y_alloc = np.zeros(self.N)
            for j in sorted_indices:
                allocation = min(self.h[j], remaining)
                y_alloc[j] = allocation
                remaining -= allocation
                if remaining <= 0:
                    break
            return y_alloc

    def matched_price_sellers(self, y_alloc):
        # Matched price: the maximum alpha among sellers with positive allocation.
        sorted_indices = np.argsort(-self.alpha)
        for j in sorted_indices:
            if y_alloc[j] > 0:
                return self.alpha[j]
        return 0

    def run(self):
        iters = 0
        diff = float('inf')
        # Gamma is fixed at R.
        Gamma = self.R  
        while iters < self.max_iter and diff > self.epsilon:
            # Compute allocation using the modified rule.
            y_alloc = self.allocate_sellers()
            ps = self.matched_price_sellers(y_alloc)
            # Our dummy buyer always bids C.
            diff = abs(self.C - ps)
            
            # Choose one seller to update.
            # For instance, update those whose alpha is not sufficiently close to C.
            candidates = [i for i in range(self.N) if y_alloc[i] > 0 and abs(self.alpha[i] - self.C) > self.epsilon]
            if not candidates:
                candidates = list(range(self.N))
            i_update = np.random.choice(candidates)
            
            # Update seller i's bid supply toward achieving a marginal cost of C.
            update_amount = self.eta * (self.C - self.alpha[i_update])
            old_h = self.h[i_update]
            self.h[i_update] = max(0, self.h[i_update] + update_amount)
            self.alpha[i_update] = self.cp(i_update, self.h[i_update])
            
            # Record state.
            self.history.append({
                'iteration': iters,
                'allocation': y_alloc.copy(),
                'matched_price_sellers': ps,
                'h': self.h.copy(),
                'alpha': self.alpha.copy(),
                'i_updated': i_update
            })
            iters += 1

# -------------------------------
# Example setup.
# Suppose we have 4 agents with cost functions, for instance:
# Let c_i(y) = a_i * (y+1)^2 for each agent.
def cost_func_factory(a):
    return lambda y: a * (y + 1)**2

# Example parameters: a = [1.0, 1.2, 0.8, 1.5]
a_params = [1.0, 1.0, 1.0, 1.5]
cost_funcs = [cost_func_factory(a) for a in a_params]

# Total resource to allocate.
R = 100.0
# Set target constant distribution cost (dummy buyer's marginal value).
C = 10.0 
# Initial bid supplies for each agent. Start with a small value.
h_init = np.ones(len(cost_funcs))

# Instantiate and run the modified auction.
auction = FixedResourceAuction(cost_funcs, R, C, h_init, epsilon=1e-3, max_iter=5000, eta=0.05)
auction.run()

final_state = auction.history[-1]
allocation = final_state['allocation']
print("Final allocation (seller side):", allocation)
print("Sum of allocation:", np.sum(allocation))
print("Final matched seller price:", final_state['matched_price_sellers'])
print("Final h (bid supplies):", final_state['h'])
print("Final alpha (bid prices):", final_state['alpha'])

# Plot convergence of matched seller price.
iters = [state['iteration'] for state in auction.history]
matched_prices = [state['matched_price_sellers'] for state in auction.history]

plt.figure(figsize=(8,4))
plt.plot(iters, matched_prices, label="Matched Seller Price")
plt.axhline(C, color='k', linestyle='dashed', label="Target Price C")
plt.xlabel("Iteration")
plt.ylabel("Matched Seller Price")
plt.title("Convergence of Matched Seller Price")
plt.legend()
plt.grid(True)
plt.show()
