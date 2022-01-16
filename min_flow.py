import numpy as np
from active_set import quadratic_problem

class min_flow (quadratic_problem):
    def __init__ (self, Q, E, b, q, l, u, verbose=False):
        m, n = E.shape
        M = np.zeros((m,m))
        super().__init__(E, b, q, 2*Q, M, l=l, u=u, verbose=verbose)

n, m = 7, 5

Q = np.zeros((n,n))
#np.fill_diagonal(Q, np.random.rand(n))
np.fill_diagonal(Q, 2)
E = np.array([[-1.,-1., 0., 0., 0., 0., 0.],
              [ 1., 0.,-1., 0., 0., 0., 0.],
              [ 0., 1., 0.,-1.,-1., 0., 0.],
              [ 0., 0., 1., 1., 0.,-1., 0.],
              [ 0., 0., 0., 0., 1., 0.,-1.]])
print(np.linalg.matrix_rank(E))

b = np.array([-1., 0., 0., 0., 0.])
q = 1 * np.ones(n)
u = 3 * np.ones(n)

qp = min_flow (Q, E, b, q, np.zeros(n), u, verbose=True)

B = np.array([True, False, True, False, True, False, True])
N = ~B

#qp.set_initial_active_set_from_factorization()
qp.solve()
