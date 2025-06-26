import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import random

class Auction:
    def __init__(self, agents, Gamma, bids, alpha=0.99, rho_bar=2, rho=0.5, epsilon=1e-4):
        """
        Initialize the Auction class for decentralized divisible resource allocation.

        Args:
            agents (List[Agent]): List of Agent objects with valuation functions.
            Gamma (float): Total amount of divisible resource to allocate.
            bids (List[Tuple[float, float]]): Initial bids for each agent, in (β, d) format.
            alpha (float): Smoothing parameter for computing constrained demand.
            rho_bar (float): Upper bound on dynamic response term for constrained demand.
            rho (float): Smoothing factor for current demand vs. allocation.
            epsilon (float): Convergence threshold for bid updates.
        """
        self.agents = agents
        self.N = len(agents)
        self.Gamma = Gamma
        self.alpha = alpha
        self.rho_bar = rho_bar
        self.rho = rho
        self.epsilon = epsilon
        self.bids = bids
        self.history = [self.bids.copy()]  # Log of all bids over time

    def valuation(self, n, x):
        """Returns the valuation u_n(x) of agent n at allocation x."""
        return self.agents[n].valuation(x)

    def marginal_valuation(self, n, x):
        """Returns the marginal valuation u_n'(x) of agent n at allocation x."""
        return self.agents[n].marginal_valuation(x)

    def allocation(self, bids):
        """
        Computes allocation vector from current bids using descending sort of β with random tie-breaking.

        Args:
            bids (List[Tuple[float, float]]): Bids as list of (β, d) pairs.

        Returns:
            np.ndarray: Allocated resource vector x for all agents.
        """
        indexed_bids = [(i, b[0], b[1], random.random()) for i, b in enumerate(bids)]
        beta_sorted = sorted(indexed_bids, key=lambda x: (-x[1], x[3]))

        x = np.zeros(self.N)
        remaining = self.Gamma
        for i, beta, d, _ in beta_sorted:
            x[i] = min(d, remaining)
            remaining -= x[i]
            if remaining <= 0:
                break
        return x

    def payment(self, bids, n):
        """
        Computes VCG payment for agent n using Clarke pivot rule.

        Args:
            bids (List[Tuple[float, float]]): Current bids.
            n (int): Agent index.

        Returns:
            float: Payment τ_n agent n must pay.
        """
        b_without_n = bids.copy()
        b_without_n[n] = (bids[n][0], 0.0)
        x_without = self.allocation(b_without_n)
        return sum(bids[m][0] * (x_without[m] - self.x[m]) for m in range(self.N) if m != n)

    def payoff(self, n, bids):
        """
        Computes payoff for agent n given current bids.

        Args:
            n (int): Agent index.
            bids (List[Tuple[float, float]]): Bid profile.

        Returns:
            float: Payoff for agent n.
        """
        x = self.allocation(bids)
        τ = self.payment(bids, n)
        return self.valuation(n, x[n]) - τ

    def constrained_demand(self, n, bids):
        """
        Computes upper bound for agent n's demand based on dynamic constraints.

        Args:
            n (int): Agent index.
            bids (List[Tuple[float, float]]): Bid profile.

        Returns:
            float: Constrained upper bound on agent n's demand.
        """
        Γc = max(0, self.Gamma - sum(d for _, d in bids))
        βn, dn = bids[n]
        m = self.find_m(n, bids)
        dm = bids[m][1] if m is not None else 0
        βm = bids[m][0] if m is not None else 0
        Φ = max(0, (βn - βm + self.rho * (dn - self.x[n]) + 0.5 * self.rho_bar * Γc)) / self.rho_bar
        upper_bound = self.x[n] + min(dm + Γc, self.alpha * Φ, (2 / self.rho_bar) * βn)
        return upper_bound

    def find_m(self, n, bids):
        """
        Finds the lowest priced winning agent (other than agent n).

        Args:
            n (int): Agent index.
            bids (List[Tuple[float, float]]): Bid profile.

        Returns:
            int: Index of agent m, or None if none found.
        """
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
        """
        Computes agent n's best response by optimizing payoff w.r.t. demand.

        Args:
            n (int): Agent index.
            bids (List[Tuple[float, float]]): Current bids.

        Returns:
            Tuple[float, float]: New bid (β, d) for agent n.
        """
        upper = self.constrained_demand(n, bids)
        lower = self.x[n]

        def neg_payoff(d):
            β = self.marginal_valuation(n, d)
            # true_beta = self.marginal_valuation(n, d)
            # budget = self.agents[n].budget

            # if budget <= 0:
            #     scaled_beta = 0.0
            # else:
            #     scaling = budget / (budget + 1e-3)
            #     scaled_beta = scaling * true_beta


            # Scaling factor — adjust if you want soft capping
            # scaling = budget / (budget + 1e-6)  # avoid divide-by-zero
            # In your budget-weighted bid scaling:
            # Instead of soft scaling, use aggressive squashing:
            # scaling = self.agents[n].budget / (self.agents[n].budget + 0.01)
            # scaled_beta = scaling * true_beta
            # effective_beta = scaling * true_beta

            temp_bids = bids.copy()
            temp_bids[n] = (β, d)
            # temp_bids[n] = (scaled_beta, d)
            return -self.payoff(n, temp_bids)

        res = minimize_scalar(neg_payoff, bounds=(lower, upper), method='bounded')
        best_d = res.x
        β_best = self.marginal_valuation(n, best_d)
        return (β_best, best_d)

    # def best_response(self, n, bids):
    #     """
    #     Computes agent n's best response by optimizing payoff with respect to demand,
    #     using budget-aware bid scaling.

    #     Args:
    #         n (int): Agent index.
    #         bids (List[Tuple[float, float]]): Current bid profile.

    #     Returns:
    #         Tuple[float, float]: New bid (β, d) for agent n.
    #     """
    #     upper = self.constrained_demand(n, bids)
    #     lower = self.x[n]

    #     budget = self.agents[n].budget

    #     # If agent is broke, skip optimization and return (0, 0)
    #     if budget <= 0:
    #         return (0.0, 0.0)

    #     def neg_payoff(d):
    #         true_beta = self.marginal_valuation(n, d)
    #         scaling = budget / (budget + 1e-3)  # aggressive scaling
    #         scaled_beta = scaling * true_beta

    #         temp_bids = bids.copy()
    #         temp_bids[n] = (scaled_beta, d)

    #         return -self.payoff(n, temp_bids)

    #     res = minimize_scalar(neg_payoff, bounds=(lower, upper), method='bounded')
    #     best_d = res.x

    #     # Recompute scaled bid for this d
    #     true_beta = self.marginal_valuation(n, best_d)
    #     scaling = budget / (budget + 1e-3)
    #     scaled_beta = scaling * true_beta

    #     return (scaled_beta, best_d)

    def compute_payments_from_delta(self, S):
        """
        Computes VCG payments based on the avoidance effort (Delta) using each agent's
        actual valuation function.

        Args:
            S (float): Total safety correction required (i.e., safety deficit).

        Returns:
            List[float]: VCG payments based on Delta externality.
        """
        # Step 1: Final credit allocation (after auction)
        c = np.array([bid[1] for bid in self.bids])
        Delta = (1 - c) * S / np.sum(1 - c)
        print(f"Δ: {[round(d, 4) for d in Delta]}")
        print("Valuations of others with agent present:",
            [round(agent.valuation(d), 4) for i, (agent, d) in enumerate(zip(self.agents, Delta)) if i != 0])

        # Then repeat with agent 1 removed (Δ_wo_1)

        N = self.N

        payments = []
        for i in range(N):
            # Step 2: Recompute allocation with agent i set to full credit (no burden)
            c_mod = c.copy()
            c_mod[i] = 1.0

            denom = np.sum(1 - c_mod)
            if denom == 0:
                Delta_wo_i = np.zeros(N)
            else:
                Delta_wo_i = (1 - c_mod) * S / denom

            # Step 3: Use agent-provided valuation functions to compute disutility
            welfare_wo_i = sum(
                -self.agents[j].valuation(Delta_wo_i[j])
                for j in range(N) if j != i
            )
            welfare_with_i = sum(
                -self.agents[j].valuation(Delta[j])
                for j in range(N) if j != i
            )

            # Step 4: VCG payment is the externality agent i imposes
            tau_i = welfare_wo_i - welfare_with_i
            payments.append(tau_i)

        return payments

    def select_next_player(self):
        """
        Selects the next agent to update their bid based on allocation status.

        Returns:
            int: Index of next player.
        """
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
        """
        Runs the auction process until convergence or max iterations.

        Args:
            max_steps (int): Maximum allowed iterations.

        Returns:
            Tuple[List[Tuple[float, float]], np.ndarray]: Final bid profile and allocation.
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
                    break
            k += 1
        return self.bids, self.allocation(self.bids)

    # def compute_payments(self):
    #     """
    #     Computes VCG payments for each agent after auction ends.

    #     Returns:
    #         List[float]: Payment τ_i for each agent.
    #     """
    #     self.x = self.allocation(self.bids)
    #     return [self.payment(self.bids, n) for n in range(self.N)]

    # def compute_payments(self):
    #     self.x = self.allocation(self.bids)
    #     payments = []

    #     for n in range(self.N):
    #         bids_wo_n = self.bids.copy()
    #         bids_wo_n[n] = (self.bids[n][0], 0.0)
    #         x_wo_n = self.allocation(bids_wo_n)

    #         print(f"\nAgent {n} removal:")
    #         for m in range(self.N):
    #             if m != n:
    #                 print(f"  Agent {m}: x_with = {self.x[m]:.4f}, x_wo_n = {x_wo_n[m]:.4f}")

    #         payment = sum(self.bids[m][0] * (x_wo_n[m] - self.x[m]) for m in range(self.N) if m != n)
    #         print(f"  Payment τ_{n} = {payment:.4f}")
    #         payments.append(payment)

    #     return payments

    def compute_payments(self):
        """
        Computes VCG payments for each agent using Clarke pivot rule with full reallocation.
        Assumes agents will reoptimize demand if another agent exits, so all Γ is consumed.
        """
        self.x = self.allocation(self.bids)
        payments = []

        for n in range(self.N):
            # 1. Remove agent n from allocation
            bids_wo_n = self.bids.copy()
            bids_wo_n[n] = (self.bids[n][0], 0.0)

            # 2. Redistribute Γ among remaining agents proportionally to their marginal value
            total_beta = sum(b[0] for i, b in enumerate(bids_wo_n) if i != n)
            if total_beta == 0:
                # No one else wants anything — payment is 0
                payments.append(0.0)
                continue

            # 3. New demand: distribute Gamma proportionally to β
            new_bids = []
            for i, (beta, _) in enumerate(bids_wo_n):
                if i == n:
                    new_bids.append((beta, 0.0))
                else:
                    new_demand = self.Gamma * (beta / total_beta)
                    new_bids.append((beta, new_demand))

            # 4. Compute new allocation
            x_wo_n = self.allocation(new_bids)

            # 5. Compute externality (benefit to others)
            externality = sum(
                self.bids[m][0] * (x_wo_n[m] - self.x[m]) for m in range(self.N) if m != n
            )
            payments.append(externality)

        return payments




    def compute_payments_vcg(self):
        """
        Computes externality-based VCG-style payments by re-running auction without each agent.

        Returns:
            List[float]: Payments for each agent.
        """
        self.x = self.allocation(self.bids)
        return [
            sum(self.bids[j][0] * (self.run_without_agent(i)[j] - self.x[j]) for j in range(self.N) if j != i)
            for i in range(self.N)
        ]

    def reset(self, agents, bids):
        """
        Resets the auction environment to a new configuration.

        Args:
            agents (List[Agent]): New agent list.
            bids (List[Tuple[float, float]]): New initial bids.
        """
        self.agents = agents
        self.N = len(agents)
        self.bids = bids.copy()
        self.history = [self.bids.copy()]
        self.x = np.zeros(self.N)

    def run_without_agent(self, remove_index):
        """
        Runs auction without a specific agent, used for computing VCG payments.

        Args:
            remove_index (int): Index of agent to exclude.

        Returns:
            np.ndarray: Full allocation vector with zero at excluded index.
        """
        agents_wo = [a for i, a in enumerate(self.agents) if i != remove_index]
        init_bids = [(a.marginal_valuation(1.0), 1.0) for a in agents_wo]
        auction_wo = Auction(agents_wo, self.Gamma, init_bids)
        _, alloc_wo = auction_wo.run()

        full_alloc = np.zeros(self.N)
        j = 0
        for i in range(self.N):
            if i == remove_index:
                continue
            full_alloc[i] = alloc_wo[j]
            j += 1
        return full_alloc

    def plot_bid_demand_over_time(self):
        """Plots demand bids over the course of the auction."""
        for i in range(self.N):
            plt.plot([step[i][1] for step in self.history], label=f'Player {i+1}')
        plt.title('Bid Demand Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Demand (d)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_bid_price_over_time(self):
        """Plots price bids (β) over the course of the auction."""
        for i in range(self.N):
            plt.plot([step[i][0] for step in self.history], label=f'Player {i+1}')
        plt.title('Bid Price Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Bid Price (β)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class Agent:
    def __init__(self, valuation_fn, marginal_valuation_fn, name="Agent"):
        """
        Agent model for auction-based resource allocation.

        Args:
            valuation_fn (Callable[[float], float]): u(x), the agent's valuation function.
            marginal_valuation_fn (Callable[[float], float]): u'(x), marginal value of allocation.
            name (str): Agent identifier.
        """
        self.valuation_fn = valuation_fn
        self.marginal_valuation_fn = marginal_valuation_fn
        self.name = name

    def valuation(self, x):
        """Returns agent's valuation at allocation x."""
        return self.valuation_fn(x)

    def marginal_valuation(self, x):
        """Returns agent's marginal valuation at allocation x."""
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