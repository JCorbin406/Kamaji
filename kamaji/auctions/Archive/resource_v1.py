import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


class Auction:
    def __init__(self, a, Gamma, alpha=0.99, rho_bar=2, rho=0.5, epsilon=1e-4):
        self.a = np.array(a)
        self.N = len(a)
        self.Gamma = Gamma
        self.alpha = alpha
        self.rho_bar = rho_bar
        self.rho = rho
        self.epsilon = epsilon
        self.bids = [(self.marginal_valuation(n, 1.0), 1.0) for n in range(self.N)]
        # self.bids = [(0.2, 0.8), (0.2, 0.8)]
        self.history = []

    def valuation(self, n, x):
    #     return 2 * self.a[n] * np.sqrt(x + 1)
        if x>=1:
            return x - x**2/2
        else:
            return 1/2

    def marginal_valuation(self, n, x):
        # return self.a[n] / np.sqrt(x + 1)
        return max(0, 1-x)

    def allocation(self, bids):
        beta_sorted = sorted([(i, b[0], b[1]) for i, b in enumerate(bids)], key=lambda x: (-x[1], x[0]))
        remaining = self.Gamma
        x = np.zeros(self.N)
        for i, beta, d in beta_sorted:
            x[i] = min(d, remaining)
            remaining -= x[i]
            if remaining <= 0:
                break
        return x

    def payment(self, bids, n):
        b_without_n = bids.copy()
        b_without_n[n] = (bids[n][0], 0.0)
        # x_with = self.allocation(bids)
        x_without = self.allocation(b_without_n)
        return sum(bids[m][0] * (x_without[m] - self.x[m]) for m in range(self.N) if m != n)

    def payoff(self, n, bids):
        x = self.allocation(bids)
        τ = self.payment(bids, n)
        return self.valuation(n, x[n]) - τ


    def constrained_demand(self, n, bids):
        Γc = max(0, self.Gamma - sum(d for _, d in bids))
        # Γc = max(0, self.Gamma - sum(d for i, (_, d) in enumerate(bids) if i != n))
        βn, dn = bids[n]
        m = self.find_m(n, bids)
        dm = bids[m][1] if m is not None else 0
        βm = bids[m][0] if m is not None else 0
        Φ = max(0, (βn - βm + self.rho * (dn - self.x[n]) + 0.5 * self.rho_bar * Γc)) / self.rho_bar
        c1 = dm + Γc
        c2 = self.alpha * Φ
        c3 = (2 / self.rho_bar) * βn
        upper_bound = self.x[n] + min(c1, c2, c3)
        return upper_bound

    def find_m(self, n, bids):
        lowest = float('inf')
        idx = None
        for i in range(self.N):
            if i == n or self.x[i] == 0:
                continue
            β = bids[i][0]
            if β < lowest or (β == lowest and (idx is None or i < idx)):
                lowest = β
                idx = i
        return idx

    # def best_response(self, n, bids):
    #     upper = self.constrained_demand(n, bids)
    #     test_ds = np.linspace(self.x[n], upper, 100)
    #     best_d = self.x[n]
    #     best_payoff = -np.inf
    #     β_best = self.marginal_valuation(n, best_d)  # Ensure default value
    #     payoffs = np.zeros((len(test_ds)))
    #     for i, d in enumerate(test_ds):
    #         β = self.marginal_valuation(n, d)
    #         temp_bids = bids.copy()
    #         temp_bids[n] = (β, d)
    #         p = self.payoff(n, temp_bids)
    #         payoffs[i] = p
    #         if p > best_payoff:
    #             best_payoff = p
    #             best_d = d
    #             β_best = β

    #     return (β_best, best_d)

    def best_response(self, n, bids):
        upper = self.constrained_demand(n, bids)
        lower = self.x[n]

        def neg_payoff(d):
            β = self.marginal_valuation(n, d)
            temp_bids = bids.copy()
            temp_bids[n] = (β, d)
            return -self.payoff(n, temp_bids)  # negative for minimization

        res = minimize_scalar(neg_payoff, bounds=(lower, upper), method='bounded')

        best_d = res.x
        β_best = self.marginal_valuation(n, best_d)
        return (β_best, best_d)



    def select_next_player(self):
        for i in range(self.N):
            β, d = self.bids[i]
            if self.x[i] < d and self.x[i] > 0:
                return i
        for i in range(self.N):
            β, d = self.bids[i]
            if self.x[i] == 0 and d > 0:
                return i
        return max(range(self.N), key=lambda i: self.bids[i][0])

    def run(self, max_steps=10000):
        k = 0
        update_log = []
        while k < max_steps:
            bids_old = self.bids.copy()
            self.x = self.allocation(self.bids)
            n = self.select_next_player()
            # n = np.random.randint(0, self.N)
            # n = 1

            self.bids[n] = self.best_response(n, self.bids)
            self.history.append(self.bids.copy())
            update_log.append(n)

            # Check termination
            if len(update_log) >= self.N:
                recent = update_log[-self.N:]
                all_updated = set(recent) == set(range(self.N))
                delta = sum(abs(self.bids[i][1] - bids_old[i][1]) + abs(self.bids[i][0] - bids_old[i][0]) for i in range(self.N))
                if all_updated and delta < self.epsilon:
                    break

            k += 1
        return self.bids, self.allocation(self.bids)
    
    def plot_bid_demand_over_time(self):
        iterations = range(len(self.history))
        for i in range(self.N):
            d_values = [step[i][1] for step in self.history]
            plt.plot(iterations, d_values, label=f'Player {i+1}')
        plt.title('Bid Demand Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Demand (d)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.xlim(0, 30)
        plt.show()

    def plot_bid_price_over_time(self):
        iterations = range(len(self.history))
        for i in range(self.N):
            beta_values = [step[i][0] for step in self.history]
            plt.plot(iterations, beta_values, label=f'Player {i+1}')
        plt.title('Bid Price Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Bid Price (β)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.xlim(0, 30)
        plt.show()

if __name__ == "__main__":
    a_values = [1.9, 2.0]
    # a_values = [1.9, 2.0, 2.1, 2.2, 2.4]
    Gamma = 1.0
    auction = Auction(a=a_values, Gamma=Gamma)
    final_bids, final_allocation = auction.run()

    S = 20
    delta = (1 - final_allocation) * S / np.sum(1-final_allocation)

    # Compute and print total social welfare
    social_welfare = sum(auction.valuation(n, final_allocation[n]) for n in range(auction.N))
    print(f"\nTotal Social Welfare: {social_welfare:.4f}")


    print("Final Allocations:")
    for i, x in enumerate(final_allocation):
        print(f"Player {i+1}: x = {x:.4f}")

    print(f"Sums: {np.sum(final_allocation):.4f}")

    print("\nFinal Bids (β_n = v'_n(d_n)):")
    for i, (beta, _) in enumerate(final_bids):
        print(f"Player {i+1}: β = {beta:.4f}")

    # print("\nFinal Delta:")
    # for i, d in enumerate(delta):
    #     print(f"Player {i+1}: Delta = {d:.4f}")

    # print(f"Sums: {np.sum(delta):.4f}")

    auction.plot_bid_demand_over_time()
    auction.plot_bid_price_over_time()