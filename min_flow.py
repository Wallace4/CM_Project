import numpy as np
import quadprog
import pickle5 as pickle
import cvxopt
from active_set import quadratic_problem

class min_flow (quadratic_problem):
    def __init__ (self, Q, E, b, q, l, u, verbose=False):
        m, n = E.shape
        M = np.zeros((m,m))
        super().__init__(E, b, q, 2*Q, M, l=l, u=u, verbose=verbose)

#n, m = 7, 5

#Q = np.zeros((n,n))
#np.fill_diagonal(Q, np.random.rand(n))
#np.fill_diagonal(Q, 2)
#E = np.array([[-1.,-1., 0., 0., 0., 0., 0.],
#              [ 1., 0.,-1., 0., 0., 0., 0.],
#              [ 0., 1., 0.,-1.,-1., 0., 0.],
#              [ 0., 0., 1., 1., 0.,-1., 0.],
#              [ 0., 0., 0., 0., 1., 0.,-1.]])
#print(np.linalg.matrix_rank(E))

#b = np.array([-2., 0., 0., 0., 0.])
#q = 1 * np.ones(n)
#u = 3 * np.ones(n)

with open("Data/problem6.pickle", "rb") as f:
    E, Q_diag, q, b, u, solution = pickle.load(f)
Q = np.diag(Q_diag)
E = E.toarray()
m, n = E.shape

qp = min_flow (Q, E, b, q, np.zeros(n), u, verbose=True)

B = np.array([ True, True, True, True, False, True, True, True, True, True])
N = ~B

#qp.set_initial_active_set_from_factorization()
try:
    qp.solve()
except:
    print("oh no")

#print(quadprog.solve_qp(Q, q, E, b, m, True))
Q = np.block([ [Q, np.zeros((n,m))],
               [np.zeros((m,n)), np.zeros((m,m))]])
q = np.concatenate((q, np.zeros(m)))
E = np.block([ [E, -np.eye(m)]])
ineqm = np.block([ [-np.eye(n+m)], [np.eye(n+m)] ])
ineqv = np.concatenate((np.zeros(n), b, u, b))
print(ineqv)
sol = cvxopt.solvers.qp(cvxopt.matrix(2*Q), cvxopt.matrix(q),
                        cvxopt.matrix(ineqm), cvxopt.matrix(ineqv),
                        cvxopt.matrix(E), cvxopt.matrix(b))

print (f"x: {sol['x']}y: {sol['y']}z: {sol['z']}s:{sol['s']}")
print (f"primal obj = {sol['primal objective']}")
print (f"dual obj = {sol['dual objective']}")
print (f"real solution: {solution}")
