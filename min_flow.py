import numpy as np
from active_set import quadratic_problem

def min_flow_to_qp (Q, E, b, q, u):
    m, n = E.shape
    A = E
    b = b
    c = q
    H = 2 * Q
#    M = np.eye(m,m)
    M = np.zeros((m,m))
    return A, b, c, H, M, u

n, m = 7, 5

Q = np.zeros((n,n))
#np.fill_diagonal(Q, np.random.rand(n))
np.fill_diagonal(Q, 1)
E = np.array([[-1.,-1., 0., 0., 0., 0., 0.],
              [ 1., 0.,-1., 0., 0., 0., 0.],
              [ 0., 1., 0.,-1.,-1., 0., 0.],
              [ 0., 0., 1., 1., 0.,-1., 0.],
              [ 0., 0., 0., 0., 1., 0.,-1.]])
print(np.linalg.matrix_rank(E))

b = np.array([ 1., -3., 0., 0., 0.])
q = 2 * np.ones(n)
u = 3 * np.ones(n)

A, b, c, H, M, u = min_flow_to_qp(Q, E, b, q, u)

qp = quadratic_problem (A, b, c, H, M, l=np.zeros(n), u=u, verbose=True)

B = np.array([True, False, True, False, True, False, True])
N = ~B

#qp.set_initial_active_set_from_factorization()
qp.solve()
