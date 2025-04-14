from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt

class DoubleSidedAuction:
    def __init__(
        self,
        valuation_funcs: List[Callable[[float], float]],
        cost_funcs: List[Callable[[float], float]],
        d_init: np.ndarray,
        h_init: np.ndarray,
        epsilon: float = 1e-4,
        max_iter: int = 1000
    ):
        self.N = len(valuation_funcs)
        self.M = len(cost_funcs)
        self.valuation_funcs = valuation_funcs
        self.cost_funcs = cost_funcs
        # Bid demands for buyers and bid supplies for sellers.
        self.d = d_init.copy()
        self.h = h_init.copy()
        # Bid prices computed as marginal valuation/cost.
        self.beta = np.array([self.vp(n, self.d[n]) for n in range(self.N)])
        self.alpha = np.array([self.cp(m, self.h[m]) for m in range(self.M)])
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.rho_bar = self.estimate_rho_bar()
        self.sigma_bar = self.estimate_sigma_bar()
        self.history = []  # Record iteration state

    def vp(self, n, x):
        delta = 1e-6
        return (self.valuation_funcs[n](x + delta) - self.valuation_funcs[n](x)) / delta

    def cp(self, m, y):
        delta = 1e-6
        return (self.cost_funcs[m](y + delta) - self.cost_funcs[m](y)) / delta

    def estimate_rho_bar(self):
        max_rho = 0
        for n in range(self.N):
            for x in np.linspace(0, 10, 100):
                delta = 1e-3
                grad = abs(self.vp(n, x + delta) - self.vp(n, x)) / delta
                max_rho = max(max_rho, grad)
        return max_rho

    def estimate_sigma_bar(self):
        max_sigma = 0
        for m in range(self.M):
            for y in np.linspace(0, 10, 100):
                delta = 1e-3
                grad = abs(self.cp(m, y + delta) - self.cp(m, y)) / delta
                max_sigma = max(max_sigma, grad)
        return max_sigma

    def allocate_buyers(self, Gamma):
        sorted_indices = np.argsort(-self.beta)
        remaining = Gamma
        x = np.zeros(self.N)
        for i in sorted_indices:
            allocation = min(self.d[i], remaining)
            x[i] = allocation
            remaining -= allocation
            if remaining <= 0:
                break
        return x

    def allocate_sellers(self, Gamma):
        sorted_indices = np.argsort(self.alpha)
        remaining = Gamma
        y = np.zeros(self.M)
        for j in sorted_indices:
            allocation = min(self.h[j], remaining)
            y[j] = allocation
            remaining -= allocation
            if remaining <= 0:
                break
        return y

    def matched_price_buyers(self, x):
        sorted_indices = np.argsort(self.beta)  # ascending order: smallest β first
        for i in sorted_indices:
            if x[i] > 0:
                return self.beta[i]
        return 0

    def matched_price_sellers(self, y):
        sorted_indices = np.argsort(-self.alpha)  # descending order: largest α first
        for j in sorted_indices:
            if y[j] > 0:
                return self.alpha[j]
        return 0

    def run(self):
        k = 0
        Gamma = min(np.sum(self.d), np.sum(self.h)) * 0.9
        eps = float('inf')
        
        while k < self.max_iter and eps > self.epsilon:
            x = self.allocate_buyers(Gamma)
            y = self.allocate_sellers(Gamma)
            pb = self.matched_price_buyers(x)
            ps = self.matched_price_sellers(y)
            Q = min(np.sum(x), np.sum(y))
            
            # Select one buyer via Equation (21) with tie breaking.
            candidate_buyers = [n for n in range(self.N) if (x[n] > 0 and x[n] < self.d[n])]
            if candidate_buyers:
                n_update = np.random.choice(candidate_buyers)
            else:
                candidate_buyers = [n for n in range(self.N) if (x[n] == 0 and self.d[n] > 0)]
                if candidate_buyers:
                    n_update = np.random.choice(candidate_buyers)
                else:
                    max_beta = np.max(self.beta)
                    candidates = [n for n in range(self.N) if self.beta[n] == max_beta]
                    n_update = np.random.choice(candidates)
            
            # Select one seller analogously.
            candidate_sellers = [m for m in range(self.M) if (y[m] > 0 and y[m] < self.h[m])]
            if candidate_sellers:
                m_update = np.random.choice(candidate_sellers)
            else:
                candidate_sellers = [m for m in range(self.M) if (y[m] == 0 and self.h[m] > 0)]
                if candidate_sellers:
                    m_update = np.random.choice(candidate_sellers)
                else:
                    min_alpha = np.min(self.alpha)
                    candidates = [m for m in range(self.M) if self.alpha[m] == min_alpha]
                    m_update = np.random.choice(candidates)
            
            # Update potential quantity.
            if (x[n_update] == 0 and self.d[n_update] > 0) or (y[m_update] == 0 and self.h[m_update] > 0):
                Gamma_next = Gamma
            else:
                Gamma_next = Q + (pb - ps) / (self.rho_bar + self.sigma_bar)
            
            # Constrained best-response update:
            increment = max(0.0, Gamma_next - Gamma)
            old_d = self.d[n_update]
            self.d[n_update] = x[n_update] + increment
            self.beta[n_update] = self.vp(n_update, self.d[n_update])
            
            old_h = self.h[m_update]
            self.h[m_update] = y[m_update] + max(0.0, Gamma_next - Gamma)
            self.alpha[m_update] = self.cp(m_update, self.h[m_update])
            
            eps = abs(Gamma_next - Gamma) + abs(self.d[n_update] - old_d) + abs(self.h[m_update] - old_h)
            Gamma = Gamma_next
            
            # Record current state, including bid profiles and bid prices.
            self.history.append({
                'iteration': k,
                'buyer_allocations': x.copy(),
                'seller_allocations': y.copy(),
                'Gamma': Gamma,
                'pb': pb,
                'ps': ps,
                'updated_buyer': n_update,
                'updated_seller': m_update,
                'd': self.d.copy(),
                'h': self.h.copy(),
                'beta': self.beta.copy(),
                'alpha': self.alpha.copy()
            })
            k += 1

def paper_example():
    # -------------------------------
    # Example 1: 6 consumers, 4 suppliers
    def example2_valuation(n):
        a_values = [2, 2.1, 2.2, 2.3, 2.4, 1.9]
        a = a_values[n]
        return lambda x: 2 * a * (x + 1) ** 0.8

    def example2_cost(m):
        b_values = [1.1, 1.5, 1.4, 1.6]
        b = b_values[m]
        return lambda y: b * (y + 1) ** 1.2

    valuation_funcs2 = [example2_valuation(n) for n in range(6)]
    cost_funcs2 = [example2_cost(m) for m in range(4)]
    # Initial bids: d^0 = h^0 = 1 for all players.
    d_init2 = np.ones(6)
    h_init2 = np.ones(4)

    dsa2 = DoubleSidedAuction(
        valuation_funcs=valuation_funcs2,
        cost_funcs=cost_funcs2,
        d_init=d_init2,
        h_init=h_init2,
        epsilon=1e-4,
        max_iter=1000
    )
    dsa2.run()

    final_state2 = dsa2.history[-1]
    print("\nFinal allocations and prices (6 Buyers, 4 Sellers Example):")
    print("Buyers' allocations:", final_state2['buyer_allocations'])
    print("Sellers' allocations:", final_state2['seller_allocations'])
    print("Potential quantity (Gamma):", final_state2['Gamma'])
    print("Matched price (buyers):", final_state2['pb'])
    print("Matched price (sellers):", final_state2['ps'])
    print("Updated buyer index:", final_state2['updated_buyer'])
    print("Updated seller index:", final_state2['updated_seller'])

    # -------------------------------
    # Define optimal values from the paper.
    x_opt = np.array([3.012, 4.122, 5.463, 7.068, 8.986, 2.104])  # Optimal consumption for buyers
    y_opt = np.array([19.873, 3.427, 5.250, 2.205])                 # Optimal supply for sellers
    p_opt = 2.424  # Optimal common marginal value

    # -------------------------------
    # Figure 4: Combined evolution of quantities and bid prices.
    # Extract iteration indices.
    iterations = [state['iteration'] for state in dsa2.history]

    # Extract bid demand/supply history.
    buyer_quantity_history = np.array([state['d'] for state in dsa2.history])  # shape: (iters, 6)
    seller_quantity_history = np.array([state['h'] for state in dsa2.history])   # shape: (iters, 4)

    # Extract bid price history.
    buyer_price_history = np.array([state['beta'] for state in dsa2.history])    # shape: (iters, 6)
    seller_price_history = np.array([state['alpha'] for state in dsa2.history])    # shape: (iters, 4)

    plt.figure(figsize=(14, 6))

    # Left panel: evolution of quantities (combined buyers and sellers).
    plt.subplot(1, 2, 1)
    for i in range(dsa2.N):
        plt.plot(iterations, buyer_quantity_history[:, i], label=f'Buyer {i+1}')
        plt.hlines(x_opt[i], iterations[0], iterations[-1], colors='k', linestyles='dashed', alpha=0.5)
    for j in range(dsa2.M):
        plt.plot(iterations, seller_quantity_history[:, j], '--', label=f'Seller {j+1}')
        plt.hlines(y_opt[j], iterations[0], iterations[-1], colors='gray', linestyles='dashed', alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Quantity")
    plt.title("Figure 4 (Left): Evolution of Agent Quantities")
    plt.legend()
    plt.grid(True)

    # Right panel: evolution of bid prices for each agent.
    plt.subplot(1, 2, 2)
    for i in range(dsa2.N):
        plt.plot(iterations, buyer_price_history[:, i], label=f'Buyer {i+1}')
    for j in range(dsa2.M):
        plt.plot(iterations, seller_price_history[:, j], '--', label=f'Seller {j+1}')
    plt.axhline(p_opt, color='k', linestyle='dashed', label='Optimal Price')
    plt.xlabel("Iteration")
    plt.ylabel("Bid Price")
    plt.title("Figure 4 (Right): Evolution of Agent Bid Prices")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Figure 5: Evolution of social welfare and matched prices.
    def compute_social_welfare(state, valuation_funcs, cost_funcs):
        x = state['buyer_allocations']
        y = state['seller_allocations']
        welfare = 0
        for n in range(len(x)):
            welfare += valuation_funcs[n](x[n])
        for m in range(len(y)):
            welfare -= cost_funcs[m](y[m])
        return welfare

    social_welfare = [compute_social_welfare(state, valuation_funcs2, cost_funcs2) for state in dsa2.history]
    matched_prices_b = [state['pb'] for state in dsa2.history]
    matched_prices_s = [state['ps'] for state in dsa2.history]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, social_welfare, 'b-', label="Social Welfare")
    plt.xlabel("Iteration")
    plt.ylabel("Social Welfare")
    plt.title("Figure 5 (Top): Evolution of Social Welfare")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iterations, matched_prices_b, 'r-', label="Matched Price (buyers)")
    plt.plot(iterations, matched_prices_s, 'g--', label="Matched Price (sellers)")
    plt.xlabel("Iteration")
    plt.ylabel("Matched Price")
    plt.title("Figure 5 (Bottom): Evolution of Matched Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cbf():
    # -------------------------------
    # Example 1: 6 consumers, 4 suppliers
    def example2_valuation(n):
        a_values = [2, 2.1, 2.2, 2.3, 2.4, 1.9]
        a = a_values[n]
        return lambda x: 2 * a * (x + 1) ** 0.8

    def example2_cost(m):
        b_values = [1.1, 1.5, 1.4, 1.6]
        b = b_values[m]
        return lambda y: (-np.log(1.0 - y + 1e-3))

    valuation_funcs2 = [example2_valuation(n) for n in range(6)]
    cost_funcs2 = [example2_cost(m) for m in range(1)]
    # Initial bids: d^0 = h^0 = 1 for all players.
    d_init2 = np.ones(6)
    h_init2 = np.ones(1)

    dsa2 = DoubleSidedAuction(
        valuation_funcs=valuation_funcs2,
        cost_funcs=cost_funcs2,
        d_init=d_init2,
        h_init=h_init2,
        epsilon=1e-4,
        max_iter=1000
    )
    dsa2.run()

    final_state2 = dsa2.history[-1]
    print("\nFinal allocations and prices (6 Buyers, 4 Sellers Example):")
    print("Buyers' allocations:", final_state2['buyer_allocations'])
    print("Sellers' allocations:", final_state2['seller_allocations'])
    print("Potential quantity (Gamma):", final_state2['Gamma'])
    print("Matched price (buyers):", final_state2['pb'])
    print("Matched price (sellers):", final_state2['ps'])
    print("Updated buyer index:", final_state2['updated_buyer'])
    print("Updated seller index:", final_state2['updated_seller'])

    # -------------------------------
    # Define optimal values from the paper.
    x_opt = np.array([3.012, 4.122, 5.463, 7.068, 8.986, 2.104])  # Optimal consumption for buyers
    y_opt = np.array([19.873, 3.427, 5.250, 2.205])                 # Optimal supply for sellers
    p_opt = 2.424  # Optimal common marginal value

    # -------------------------------
    # Figure 4: Combined evolution of quantities and bid prices.
    # Extract iteration indices.
    iterations = [state['iteration'] for state in dsa2.history]

    # Extract bid demand/supply history.
    buyer_quantity_history = np.array([state['d'] for state in dsa2.history])  # shape: (iters, 6)
    seller_quantity_history = np.array([state['h'] for state in dsa2.history])   # shape: (iters, 4)

    # Extract bid price history.
    buyer_price_history = np.array([state['beta'] for state in dsa2.history])    # shape: (iters, 6)
    seller_price_history = np.array([state['alpha'] for state in dsa2.history])    # shape: (iters, 4)

    plt.figure(figsize=(14, 6))

    # Left panel: evolution of quantities (combined buyers and sellers).
    plt.subplot(1, 2, 1)
    for i in range(dsa2.N):
        plt.plot(iterations, buyer_quantity_history[:, i], label=f'Buyer {i+1}')
        plt.hlines(x_opt[i], iterations[0], iterations[-1], colors='k', linestyles='dashed', alpha=0.5)
    for j in range(dsa2.M):
        plt.plot(iterations, seller_quantity_history[:, j], '--', label=f'Seller {j+1}')
        plt.hlines(y_opt[j], iterations[0], iterations[-1], colors='gray', linestyles='dashed', alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Quantity")
    plt.title("Figure 4 (Left): Evolution of Agent Quantities")
    plt.legend()
    plt.grid(True)

    # Right panel: evolution of bid prices for each agent.
    plt.subplot(1, 2, 2)
    for i in range(dsa2.N):
        plt.plot(iterations, buyer_price_history[:, i], label=f'Buyer {i+1}')
    for j in range(dsa2.M):
        plt.plot(iterations, seller_price_history[:, j], '--', label=f'Seller {j+1}')
    plt.axhline(p_opt, color='k', linestyle='dashed', label='Optimal Price')
    plt.xlabel("Iteration")
    plt.ylabel("Bid Price")
    plt.title("Figure 4 (Right): Evolution of Agent Bid Prices")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # -------------------------------
    # Figure 5: Evolution of social welfare and matched prices.
    def compute_social_welfare(state, valuation_funcs, cost_funcs):
        x = state['buyer_allocations']
        y = state['seller_allocations']
        welfare = 0
        for n in range(len(x)):
            welfare += valuation_funcs[n](x[n])
        for m in range(len(y)):
            welfare -= cost_funcs[m](y[m])
        return welfare

    social_welfare = [compute_social_welfare(state, valuation_funcs2, cost_funcs2) for state in dsa2.history]
    matched_prices_b = [state['pb'] for state in dsa2.history]
    matched_prices_s = [state['ps'] for state in dsa2.history]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, social_welfare, 'b-', label="Social Welfare")
    plt.xlabel("Iteration")
    plt.ylabel("Social Welfare")
    plt.title("Figure 5 (Top): Evolution of Social Welfare")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iterations, matched_prices_b, 'r-', label="Matched Price (buyers)")
    plt.plot(iterations, matched_prices_s, 'g--', label="Matched Price (sellers)")
    plt.xlabel("Iteration")
    plt.ylabel("Matched Price")
    plt.title("Figure 5 (Bottom): Evolution of Matched Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # paper_example()
    cbf()