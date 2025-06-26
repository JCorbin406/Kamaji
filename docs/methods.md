# Methods

Kamaji supports research in safety-critical multi-agent systems using modern control techniques, including:

---

## 🛡 Control Barrier Functions (CBFs)

Kamaji uses Zeroing Control Barrier Functions to enforce safety constraints:

- Pairwise agent safety is encoded with functions `h_ij(x)`
- A global barrier function `H(x)` is formed via log-sum-exp
- When `Aû < b`, agents cooperatively adjust control inputs

---

## 💰 Auction-Based Fairness

Agents bid to reduce their safety burden using an internal credit mechanism:

- Agents submit bids for avoidance credit
- The credit determines their control effort contribution (`∆i`)
- A VCG-style payment scheme encourages truthful bidding

---

## 🧠 Nominal Control & Correction

Each agent has:
- A **nominal controller** (e.g., PID, geometric)
- A **correction mechanism** (e.g., CBF via auction or QP)

Correction only activates if safety constraints are violated.

---

## 🔬 Applications

- Urban Air Mobility (UAM)
- Formation control
- Distributed collision avoidance
- Multi-agent negotiation and fairness

