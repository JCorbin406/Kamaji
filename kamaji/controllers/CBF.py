import logging
import numpy as np
from sympy import Matrix, lambdify, simplify
import matplotlib.pyplot as plt
import networkx as nx
from qpsolvers import solve_qp

logger = logging.getLogger(__name__)


class QPInfeasibleError(RuntimeError):
    """Raised when the CBF safety QP is infeasible."""
    pass


class CBFFilterResult:
    """Result of a CBF control filtering operation.

    Attributes:
        control: The filtered control vector (u_nom if QP was infeasible and fallback used).
        feasible: Whether the QP was feasible.
    """
    __slots__ = ("control", "feasible")

    def __init__(self, control: np.ndarray, feasible: bool):
        self.control = control
        self.feasible = feasible


class CBFSystem:
    """Control Barrier Function (CBF) safety filter system.

    Manages a collection of CBF constraints and enforces them on nominal
    control inputs via a Quadratic Program (QP). The pipeline is:

    1. **Define constraints** — Call ``add_cbf()`` with symbolic SymPy
       expressions for the barrier function *h*, drift *f*, actuation *g*,
       and an extended class-K function *alpha*.
    2. **Symbolic differentiation** — ``add_cbf()`` automatically computes
       the Lie derivatives L_f h and L_g h, then lambdifies them for fast
       numerical evaluation.
    3. **Evaluate & filter** — ``filter_controls()`` evaluates all
       constraints at the current state, assembles the QP, and solves for
       the minimally-modified safe control:

       .. math::

           u^* = \\arg\\min_u \\|u - u_{nom}\\|^2
           \\quad \\text{s.t.} \\quad L_f h + L_g h \\, u + \\alpha(h) \\geq 0

    Args:
        cbf_dict: Optional dictionary of CBF specifications to add at
            construction. Each value must contain keys ``agents``,
            ``state_vars``, ``h_expr``, ``f_expr``, ``g_expr``,
            ``alpha_func``.
        on_infeasible: Optional callback invoked when the QP is infeasible.
            Signature: ``(state_values, u_nom) -> CBFFilterResult``.
            If ``None``, falls back to returning ``u_nom`` with a warning.

    Attributes:
        cbf_terms: Dictionary mapping CBF IDs to their symbolic/numeric data.
        agent_index: Dictionary mapping agent IDs to the list of CBF IDs
            that involve that agent.
    """

    def __init__(self, cbf_dict=None, on_infeasible=None):
        self.cbf_terms = {}
        self.agent_index = {}
        self.on_infeasible = on_infeasible

        if cbf_dict:
            for cbf_id, spec in cbf_dict.items():
                self.add_cbf(
                    cbf_id=cbf_id,
                    agents=spec['agents'],
                    state_vars=spec['state_vars'],
                    h_expr=spec['h_expr'],
                    f_expr=spec['f_expr'],
                    g_expr=spec['g_expr'],
                    alpha_func=spec['alpha_func']
                )

    def add_cbf(self, cbf_id, agents, state_vars, h_expr, f_expr, g_expr, alpha_func):
        """Register a new CBF constraint.

        Performs symbolic differentiation to compute grad(h), L_f h, and
        L_g h, then stores lambdified (numerical) versions for fast
        evaluation during simulation.

        Args:
            cbf_id: Unique string identifier for this constraint
                (e.g. ``"cbf_01"``).
            agents: List of agent IDs involved in this constraint.
            state_vars: Ordered list of SymPy symbols representing the
                state variables that appear in h, f, and g (e.g.
                ``[x0, y0, x1, y1]``).
            h_expr: SymPy scalar expression for the barrier function h(x).
                Must be positive in the safe set.
            f_expr: SymPy column Matrix for the drift dynamics f(x)
                (dimension must match ``state_vars``).
            g_expr: SymPy Matrix for the actuation matrix g(x)
                (rows = len(state_vars), cols = total control dimension).
            alpha_func: Extended class-K function applied to h. Must be a
                callable ``alpha(h_val) -> float`` (e.g.
                ``lambda h: 2.0 * h`` for a linear class-K function).
        """
        grad_h = h_expr.diff(Matrix(state_vars)).T.doit()
        LfH = (grad_h * f_expr).doit().as_mutable()
        LgH = (grad_h * g_expr).doit().as_mutable()

        # Ensure scalar expressions and avoid nested structures
        LfH_exprs = [simplify(LfH[i]) for i in range(LfH.shape[0])]
        LgH_exprs = [simplify(LgH[0, j]) for j in range(LgH.shape[1])]

        self.cbf_terms[cbf_id] = {
            'id': cbf_id,
            'agents': agents,
            'vars': state_vars,
            'h_expr': h_expr,
            'h': lambdify(state_vars, h_expr, modules='numpy'),
            'grad_h': grad_h,
            'LfH': lambdify(state_vars, LfH_exprs, modules='numpy'),
            'LgH': lambdify(state_vars, LgH_exprs, modules='numpy'),
            'f_expr': f_expr,
            'g_expr': g_expr,
            'alpha': alpha_func
        }

        for aid in agents:
            self.agent_index.setdefault(aid, []).append(cbf_id)

    def evaluate_single_constraint(self, cbf_id, state_values):
        """Evaluate one CBF constraint at the given state.

        Computes the constraint row for the QP inequality ``-L_g h * u <= L_f h + alpha(h)``.

        Args:
            cbf_id: ID of the CBF constraint to evaluate.
            state_values: Dictionary mapping state variable *names* (strings)
                to their current numerical values.

        Returns:
            Tuple of ``(A_row, b_elem, h_val)`` where:

            - ``A_row`` — shape ``(1, n_controls)``, the ``-L_g h`` row for the QP.
            - ``b_elem`` — scalar, the ``-(L_f h + alpha(h))`` right-hand side.
            - ``h_val`` — scalar, current value of the barrier function.
        """
        term = self.cbf_terms[cbf_id]
        vals = [state_values[str(v)] for v in term['vars']]
        h_val = term['h'](*vals)
        LfH_val = np.array(term['LfH'](*vals)).flatten()
        LgH_val = np.array(term['LgH'](*vals)).reshape(1, -1)
        alpha_val = term['alpha'](h_val)
        return LgH_val, -LfH_val - alpha_val, h_val

    def evaluate_constraints(self, state_values):
        """Evaluate all CBF constraints and stack them into QP matrices.

        Args:
            state_values: Dictionary mapping state variable names to values.

        Returns:
            Tuple of ``(A, b)`` where ``A`` has shape ``(n_constraints, n_controls)``
            and ``b`` has shape ``(n_constraints,)``, representing the inequality
            ``-A u <= b`` (equivalently ``A u + b >= 0``).
        """
        A_list, b_list = [], []
        for cbf_id in self.cbf_terms:
            A, b, _ = self.evaluate_single_constraint(cbf_id, state_values)
            A_list.append(A)
            b_list.append(b)
        return np.vstack(A_list), np.hstack(b_list)

    def filter_controls(self, state_values, u_nom, u_bounds=None, mode="all"):
        """Filter nominal controls through CBF safety constraints via QP.

        Solves the min-norm QP:

            min_u  ||u - u_nom||^2
            s.t.   L_f h + L_g h * u + alpha(h) >= 0   (for each CBF)

        Args:
            state_values: Dict mapping state variable names to current values.
            u_nom: Nominal (desired) control vector.
            u_bounds: Optional ``(lb, ub)`` tuple for element-wise control bounds.
            mode: Constraint evaluation mode.

        Returns:
            CBFFilterResult with the (possibly filtered) control and feasibility flag.

        Raises:
            QPInfeasibleError: If the QP is infeasible and no on_infeasible callback
                is set and no fallback behavior is desired. By default, falls back
                to u_nom with a warning.
        """
        A, b = self.evaluate_constraints(state_values)
        H = np.eye(len(u_nom))
        f = -u_nom
        lb, ub = None, None
        if u_bounds:
            lb, ub = u_bounds
        u_star = solve_qp(H, f, G=-A, h=-b, lb=lb, ub=ub, solver="cvxopt")
        if u_star is None:
            logger.warning("CBF QP infeasible — safety constraint cannot be satisfied.")
            if self.on_infeasible is not None:
                return self.on_infeasible(state_values, u_nom)
            return CBFFilterResult(control=u_nom, feasible=False)
        return CBFFilterResult(control=u_star, feasible=True)

    def visualize_agent_links(self):
        """Display a graph of which agents share CBF constraints.

        Draws an undirected graph where nodes are agents and edges connect
        agent pairs that share at least one pairwise CBF constraint.
        """
        G = nx.Graph()
        for cbf in self.cbf_terms.values():
            agents = cbf['agents']
            if len(agents) == 2:
                G.add_edge(agents[0], agents[1])
            elif len(agents) == 1:
                G.add_node(agents[0])
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
        plt.title("CBF Agent Constraint Graph")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    from sympy import symbols, Matrix, sqrt

    x0, y0 = symbols('x0 y0')
    x1, y1 = symbols('x1 y1')
    x2, y2 = symbols('x2 y2')

    agent_states = {
        0: [x0, y0],
        1: [x1, y1],
        2: [x2, y2]
    }

    all_vars = agent_states[0] + agent_states[1] + agent_states[2]
    f = Matrix([0.0] * len(all_vars))
    g = Matrix.eye(len(all_vars))
    alpha = lambda h: 2.0 * h

    cbf_sys = CBFSystem()

    # Helper to extract f and g for selected variables
    def extract_fg(state_vars, all_state_vars, f_full, g_full):
        indices = [all_state_vars.index(v) for v in state_vars]
        f_sub = f_full.extract(indices, [0])
        g_sub = g_full.extract(indices, list(range(g_full.shape[1])))
        return f_sub, g_sub

    # Manually define pairwise CBFs
    h01 = sqrt((x0 - x1)**2 + (y0 - y1)**2) - 1.0
    h02 = sqrt((x0 - x2)**2 + (y0 - y2)**2) - 1.0

    f01, g01 = extract_fg([x0, y0, x1, y1], all_vars, f, g)
    f02, g02 = extract_fg([x0, y0, x2, y2], all_vars, f, g)

    cbf_sys.add_cbf("cbf_01", [0, 1], [x0, y0, x1, y1], h01, f01, g01, alpha)
    cbf_sys.add_cbf("cbf_02", [0, 2], [x0, y0, x2, y2], h02, f02, g02, alpha)

    state_values = {
        'x0': 0.0, 'y0': 0.0,
        'x1': 0.5, 'y1': 0.0,
        'x2': 0.4, 'y2': 0.0
    }

    u_nom = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    u_filtered = cbf_sys.filter_controls(state_values, u_nom, mode="all")
    print("Filtered control output:\n", u_filtered)

    cbf_sys.visualize_agent_links()
