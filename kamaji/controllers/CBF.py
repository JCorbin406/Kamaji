import numpy as np
from sympy import Matrix, lambdify, simplify
import matplotlib.pyplot as plt
import networkx as nx
from qpsolvers import solve_qp


class CBFSystem:
    def __init__(self, cbf_dict=None):
        self.cbf_terms = {}
        self.agent_index = {}

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
        term = self.cbf_terms[cbf_id]
        vals = [state_values[str(v)] for v in term['vars']]
        h_val = term['h'](*vals)
        LfH_val = np.array(term['LfH'](*vals)).flatten()
        LgH_val = np.array(term['LgH'](*vals)).reshape(1, -1)
        alpha_val = term['alpha'](h_val)
        return LgH_val, -LfH_val - alpha_val, h_val

    def evaluate_constraints(self, state_values):
        A_list, b_list = [], []
        for cbf_id in self.cbf_terms:
            A, b, _ = self.evaluate_single_constraint(cbf_id, state_values)
            A_list.append(A)
            b_list.append(b)
        return np.vstack(A_list), np.hstack(b_list)

    def filter_controls(self, state_values, u_nom, u_bounds=None, mode="all"):
        A, b = self.evaluate_constraints(state_values)
        H = np.eye(len(u_nom))
        f = -u_nom
        lb, ub = None, None
        if u_bounds:
            lb, ub = u_bounds
        u_star = solve_qp(H, f, G=-A, h=-b, lb=lb, ub=ub, solver="cvxopt")
        if u_star is None:
            print("Warning: QP infeasible — returning nominal control.")
            return u_nom
        return u_star

    def visualize_agent_links(self):
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