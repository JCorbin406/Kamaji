import sympy as sp

# Define state variables x1, x2, ..., x6 as symbols
x1, x2, x3, x4, x5, x6 = sp.symbols('x1 x2 x3 x4 x5 x6')
x = sp.Matrix([x1, x2, x3, x4, x5, x6])

# Define parameters (e.g., xc, yc, zc for h)
xc, yc, zc, r = sp.symbols('xc yc zc r')

# Define h(x)
h = sp.sqrt((x1 - xc)**2 + (x2 - yc)**2 + (x3 - zc)**2) - r

# Define f(x) as a symbolic matrix
f = sp.Matrix([x4, x5, x6, 0, 0, 0])

# Define g(x) as a symbolic matrix
g = sp.Matrix([0, 0, 0, 1, 1, 1])

# Compute the gradient of h with respect to x
grad_h = sp.Matrix([sp.diff(h, var) for var in x])

# First Lie derivative: L_f h(x) = grad(h) · f
Lf_h = grad_h.dot(f)

# Compute the gradient of L_f h with respect to x
grad_Lf_h = sp.Matrix([sp.diff(Lf_h, var) for var in x])

# Second Lie derivative: L_f^2 h(x) = grad(L_f h) · f
Lf2_h = grad_Lf_h.dot(f)

# Lie derivative along g of L_f h(x): L_g L_f h(x) = grad(L_f h) · g
Lg_Lf_h = grad_Lf_h.dot(g)

# print(sp.latex(grad_h))