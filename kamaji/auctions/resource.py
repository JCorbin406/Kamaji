import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import random


class Auction:
    def __init__(self, agents, Gamma, bids, alpha=0.99, rho_bar=2, rho=0.5, epsilon=1e-4):
        """Initialize the auction environment.

        Args:
            agents (List[Agent]): List of agent objects, each with their own valuation functions.
            Gamma (float): Total amount of divisible resource.
            bids (List[Tuple[float, float]]): Initial bids from agents in the form (β, d).
            alpha (float): Smoothing parameter for demand updates.
            rho_bar (float): Scaling factor used in constrained demand calculation.
            rho (float): Parameter for demand smoothing.
            epsilon (float): Convergence threshold.
        """
        self.agents = agents
        self.N = len(agents)
        self.Gamma = Gamma
        self.alpha = alpha
        self.rho_bar = rho_bar
        self.rho = rho
        self.epsilon = epsilon
        self.bids = bids
        self.history = []
        self.history.append(self.bids.copy())

    def valuation(self, n, x):
        """Evaluate the valuation function for agent n at allocation x."""
        return self.agents[n].valuation(x)

    def marginal_valuation(self, n, x):
        """Evaluate the marginal valuation function for agent n at allocation x."""
        return self.agents[n].marginal_valuation(x)

    def allocation(self, bids):
        """Compute allocation based on sorted bid prices (descending).

        Args:
            bids (List[Tuple[float, float]]): Bids as (β, d) pairs.

        Returns:
            np.ndarray: Allocated quantities to each agent.
        """
        # beta_sorted = sorted([(i, b[0], b[1]) for i, b in enumerate(bids)], key=lambda x: (-x[1], x[0]))
        
        """Commenting out for debugging"""
        # Add a random tie-breaker to each agent
        indexed_bids = [(i, b[0], b[1], random.random()) for i, b in enumerate(bids)]
        # Sort by descending bid price β, then by random tiebreaker
        beta_sorted = sorted(indexed_bids, key=lambda x: (-x[1], x[3]))

        """2nd attempt. Prior one works I believe"""
        # indexed_bids = [(i, b[0], b[1]) for i, b in enumerate(bids)]
        # beta_sorted = sorted(indexed_bids, key=lambda x: (-x[1], x[0]))  # tie-break by index

        remaining = self.Gamma
        x = np.zeros(self.N)
        for i, beta, d, _ in beta_sorted:
            x[i] = min(d, remaining)
            remaining -= x[i]
            if remaining <= 0:
                break

        # allocation_threshold = 1e-4  # or tighter if needed
        # for i, beta, d, _ in beta_sorted:
        #     if remaining < allocation_threshold:
        #         break
        #     x[i] = min(d, remaining)
        #     remaining -= x[i]

        return x

    def payment(self, bids, n):
        """Calculate payment for agent n using Clarke pivot rule."""
        b_without_n = bids.copy()
        b_without_n[n] = (bids[n][0], 0.0)
        x_without = self.allocation(b_without_n)
        return sum(bids[m][0] * (x_without[m] - self.x[m]) for m in range(self.N) if m != n)

    def payoff(self, n, bids):
        """Calculate agent n's payoff under given bids."""
        x = self.allocation(bids)
        τ = self.payment(bids, n)
        return self.valuation(n, x[n]) - τ

    def constrained_demand(self, n, bids):
        """Compute upper bound on demand for agent n based on dynamic constraints."""
        Γc = max(0, self.Gamma - sum(d for _, d in bids))
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
        """Find the lowest-priced winning agent (excluding agent n)."""
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

    def best_response(self, n, bids):
        """Compute agent n's best response (optimal β, d) given current bids."""
        upper = self.constrained_demand(n, bids)
        lower = self.x[n]

        def neg_payoff(d):
            β = self.marginal_valuation(n, d)
            temp_bids = bids.copy()
            temp_bids[n] = (β, d)
            return -self.payoff(n, temp_bids)

        res = minimize_scalar(neg_payoff, bounds=(lower, upper), method='bounded')
        best_d = res.x
        β_best = self.marginal_valuation(n, best_d)
        return (β_best, best_d)

    def select_next_player(self):
        """Determine the next agent to update its bid."""
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
        """Run the auction until convergence or max_steps.

        Returns:
            Tuple[List[Tuple[float, float]], np.ndarray]: Final bids and allocations.
        """
        k = 0
        update_log = []
        while k < max_steps:
            bids_old = self.bids.copy()
            self.x = self.allocation(self.bids)
            n = self.select_next_player()
            self.bids[n] = self.best_response(n, self.bids)

            self.history.append(self.bids.copy())
            update_log.append(n)

            if len(update_log) >= self.N:
                recent = update_log[-self.N:]
                all_updated = set(recent) == set(range(self.N))
                delta = sum(abs(self.bids[i][1] - bids_old[i][1]) + abs(self.bids[i][0] - bids_old[i][0]) for i in range(self.N))
                if all_updated and delta < self.epsilon:
                    # print('Converged')
                    break
            k += 1
        return self.bids, self.allocation(self.bids)
    
    def compute_payments(self):
        """Compute Clarke pivot payments τ_i for each agent after final allocation."""
        self.x = self.allocation(self.bids)  # Ensure allocation is fresh
        payments = []
        for n in range(self.N):
            τ_n = self.payment(self.bids, n)
            payments.append(τ_n)
        return payments

    def compute_payments_vcg(self):
        """
        Compute Clarke-pivot (VCG-style) payments after the auction has converged.

        Returns:
            List[float]: Payment τ_i for each agent i.
        """
        self.x = self.allocation(self.bids)  # update self.x to current allocation
        payments = []
        for i in range(self.N):
            x_without = self.run_without_agent(i)
            externality = sum(
                self.bids[j][0] * (x_without[j] - self.x[j])
                for j in range(self.N) if j != i
            )
            payments.append(externality)
        return payments

    def reset(self, agents, bids):
        """
        Reset the auction to a new set of agents and initial bids.

        Args:
            agents (List[Agent]): New or original list of Agent objects.
            bids (List[Tuple[float, float]]): Initial (β, d) bids for each agent.
        """
        self.agents = agents
        self.N = len(agents)
        self.bids = bids.copy()
        self.history = [self.bids.copy()]
        self.x = np.zeros(self.N)  # reset allocation


    def run_without_agent(self, remove_index):
        """
        Run the auction with agent `remove_index` removed.
        Returns the full-length allocation (with 0 for the removed agent).
        """
        # Subset of agents
        agents_wo = [a for i, a in enumerate(self.agents) if i != remove_index]

        # Initialize fresh marginal bids (not fixed, they'll update in run())
        init_bids = [(agent.marginal_valuation(1.0), 1.0) for agent in agents_wo]

        # Create and run new auction with subset
        auction_wo = Auction(agents=agents_wo, Gamma=self.Gamma, bids=init_bids)
        final_bids_wo, alloc_wo = auction_wo.run()

        # Pad result back to original length
        full_alloc = np.zeros(self.N)
        j = 0
        for i in range(self.N):
            if i == remove_index:
                continue
            full_alloc[i] = alloc_wo[j]
            j += 1
        return full_alloc


    def plot_bid_demand_over_time(self):
        """Visualize the evolution of demand bids over iterations."""
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
        plt.show()

    def plot_bid_price_over_time(self):
        """Visualize the evolution of price bids over iterations."""
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
        plt.show()


class Agent:
    def __init__(self, valuation_fn, marginal_valuation_fn, name="Agent"):
        """Agent class for holding valuation and marginal valuation logic.

        Args:
            valuation_fn (Callable[[float], float]): Valuation function u(x)
            marginal_valuation_fn (Callable[[float], float]): Marginal valuation function u'(x)
            name (str, optional): Agent identifier. Defaults to "Agent".
        """
        self.valuation_fn = valuation_fn
        self.marginal_valuation_fn = marginal_valuation_fn
        self.name = name

    def valuation(self, x):
        """Evaluate this agent's valuation at allocation x."""
        return self.valuation_fn(x)

    def marginal_valuation(self, x):
        """Evaluate this agent's marginal valuation at allocation x."""
        return self.marginal_valuation_fn(x)

if __name__ == "__main__":
    """2 equal agents example."""
    def v(x): return x - x**2/2 if x >= 1 else 0.5
    def v_prime(x): return max(0, 1 - x)
    agents = [Agent(valuation_fn=v, marginal_valuation_fn=v_prime, name=f"Player {i+1}") for i in range(2)]
    initial_bids = [(0.2, 0.8), (0.2, 0.8)]
    Gamma = 1.0

    """5 heterogeneous agents example."""
    # a_values = [1.9, 2.0, 2.1, 2.2, 2.4]
    # agents = []
    # for i, a in enumerate(a_values):
    #     valuation_fn = lambda x, a=a: 2 * a * np.sqrt(x + 1)
    #     marginal_fn = lambda x, a=a: a / np.sqrt(x + 1)
    #     agents.append(Agent(valuation_fn, marginal_fn, name=f"Player {i+1}"))
    # initial_bids = [(agent.marginal_valuation(1.0), 1.0) for agent in agents]
    # Gamma = 20.0
    
    auction = Auction(agents=agents, Gamma=Gamma, bids=initial_bids)

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