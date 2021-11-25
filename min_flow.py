import numpy as np
from active_set import quadratic_problem

def min_flow_to_qp (Q, E, b, q, u):
    m, n = E.shape
    A = E
    b = b
    c = q
    H = 2 * Q
    M = np.eye(m,m)
    return A, b, c, H, M

n = m = 6

Q = np.zeros((n,m))
np.fill_diagonal(Q, np.random.rand(n))
E = np.array([[0., 1., 1., 0., 0., 0.],
              [0., 0., 1., 1., 0., 0.],
              [0., 1., 0., 1., 0., 0.],
              [0., 0., 1., 0., 0., 1.],
              [0., 0., 0., 1., 0., 1.],
              [0., 0., 0., 0., 0., 0.]])
b = 2 * np.ones(m)
q = np.ones(n)
u = 3 * np.ones(n)

A, b, c, H, M = min_flow_to_qp(Q, E, b, q, u)

qp = quadratic_problem (A, b, c, H, M, verbose=True)

B = np.array([True, True, False, False, True, False])
N = ~B

qp.primal_first_strategy(B, N)
