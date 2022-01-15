#!bin/usr/python3
# coding=utf-8

import numpy as np
from scipy.linalg import lu, solve, ldl, cholesky, lstsq
from scipy.sparse.linalg import splu, spsolve
import math
import logging
import sys
import argparse

def solve_plin (A, b):
#    A_i = np.linalg.pinv (A)
#    P, L, U = lu (A)
#    c = np.linalg.pinv(L) @ b
#    x = np.linalg.pinv(U) @ c
#    print(f"{x}\n{np.linalg.inv(P)@x}")
    return spsolve(A, b)

def norm_2 (exp):
    norm = np.linalg.norm(exp, 2)
    print(norm)
    return pow(norm, 2)

class quadratic_problem:
    """! The quadratic problem class
    
    Define the base class used to resolve the active set algorithms for quadratic problems
    """

    @staticmethod
    def __check_shape(x, dim=None, varname=""):
        """! Static methot for checking the shape of the matrixes and vectors

        @param x the np.array that need to be checked
        @param dim the dimension that the vector should have, defaults to "None"
        @param varname the name of the variable, defaults to ""

        @return x if the dim is right
        """
        if isinstance(x, np.ndarray):
            if dim == None or x.shape == dim:
                return x
            else:
                raise TypeError(f'<{varname}> has shape <{x.shape}> expected <{dim}>')
        else:
            raise TypeError(f'<{varname}> is not {type({np.ndarray})}')

    def __init__(self, A, b, c, H, M, l=0, u=np.inf, tol=1e-8, verbose=False):
        """! The Quadratic Problem Class initializer

        @param A matrix R(m,n)
        @param b vector R(m)
        @param c vector R(n)
        @param H Positive Symmetric Hessian Matrix R(n,n)
        @param M Positive Symmetric Hessian Matrix R(m,m)
        @param q vector R(n), default to np.zeros(n)
        @param r vector R(n), default to np.zeros(n)
        @param tol maximum tolerance for operations, defaults to 1e-8
        @param verbose if we want to log to the stdout too or not, default to False
        """
        self.logger = logging.getLogger('execution_logger')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler ("execution.log", mode='w')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        
        if verbose:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
        self.logger.info('Class created with success and started the logger')
        self.tol = tol
        
        # shape of A is assumed to be correct
        self.A = self.__check_shape(A, varname="A")
        m, n = A.shape

        self.b = self.__check_shape(b, dim=(m,), varname="b")
        self.c = self.__check_shape(c, dim=(n,), varname="c")
        self.H = self.__check_shape(H, dim=(n, n), varname="H")
        self.M = self.__check_shape(M, dim=(m, m), varname="M")

#        if (np.all((M == 1))): #controllo se la matrice M è 0. Se si dobbiamo usare le variabili slack
            #self.s = self.b #dobbiamo in realtà usare y come se fosse s, in questo modo noi avremmo s come variabile, e dobbiamo solo farla rientrare entro i vincoli, visto che è l'unica variabile non vincolata.
            #self.b = np.zeros((m,)) #b in questo caso sarà 0, mentre s sono le variabili slack che per forza devono avere il valore di b, dato che sono tutti vincoli di uguaglianza
#            rank = np.linalg.matrix_rank(np.block([A, np.eye(m)]), tol=self.tol)
            #TODO implementare questo cambiamento ovunque tipo
        #else:
        rank = np.linalg.matrix_rank(np.block([A, -M]), tol=self.tol)
        assert (rank == m), f"Not full row rank matrix, {rank} != {m}"

        #init lower bound shifts and z
        self.l = self.__check_shape(l, dim=(n,), varname="l") if l is not -np.inf else -np.inf
        self.q_l = np.zeros(n) #q e r sono soltando gli shift, quindi interni alla costruzione del problema
        self.r_l = np.zeros(n)
        self.z_l = np.zeros(n)
        self.dz_l = np.zeros(n)

        #init upper bound shift and z
        self.u = self.__check_shape(u, dim=(n,), varname="u") if u is not np.inf else np.inf
        self.q_u = np.zeros(n)
        self.r_u = np.zeros(n) 
        self.z_u = np.zeros(n)
        self.dz_u = np.zeros(n)

        # init solutions
        self.x = np.zeros(n)
        self.y = np.zeros(m) 
        self.z = np.zeros(n) #tmp bad
        
        # init deltas
        self.dx = np.zeros(n)
        self.dy = np.zeros(m)
        self.dz = np.zeros(n)

        # init B, N (temporary)
        self.B = np.full(n, True)
        self.N = np.full(n, False)

    #deprecated
    def set_initial_solution(self, x, y, z):
        """! Function that set the initial solution of the problem
        
        @param x initial values of the x vector R(n)
        @param y initial values of the y vector R(m)
        @param z initial values of the z vector R(n)
        """
        m, n = self.A.shape
        self.x[:] = self.__check_shape(x, dim=(n,), varname="x")
        self.y[:] = self.__check_shape(y, dim=(m,), varname="y")
        self.z[:] = self.__check_shape(z, dim=(n,), varname="z")
        self.logger.info(f"Successfully set the initial solutions:\nx:\n{self.x}\ny:\n{self.y}\nz:\n{self.z}")

    #deprecated
    def set_initial_active_set(self, B, N):
        """! Function that set the initial active set of the problem
        
        @param B initial values of B boolean vector R(n)
        @param N initial values of N boolean vector R(n)
        """
        assert all(np.logical_xor(B, N)), "Sets are not valid. |Union| should be n and |Intersection| should be 0"
        n = self.A.shape[1]
        self.B[:] = self.__check_shape(B, dim=(n,), varname="B")
        self.N[:] = self.__check_shape(N, dim=(n,), varname="N")
        self.logger.info(f"Successfully set the initial active set:\nB:\n{self.B}\nN:\n{self.N}")

    #stuff che non funziona :/
    def set_initial_active_set_from_factorization (self): #metodo che non funziona bene ancora
        K = np.block([
            [self.H,  self.A.T],
            [self.A, -self.M  ]
        ])
        L, D, P = ldl(K)

        last_ele = -1
        for ele in P:
            i = ele+1
            if (last_ele > ele):
                break
        print(P)
        for ele in P[i:]: #di default B è tutto vero, e N è tutto falso
            print(ele)
            self.B[ele] = False
            self.N[ele] = True
        assert all(np.logical_xor(B,N)), "Sets are not valid. |Union| should be n and |Intersection| should be 0"
        self.logger.info(f"Successfully set the initial active set:\nB:\n{self.B}\nN:\n{self.N}")
        print(P)

    #attualmente utilizzata
    def set_initial_active_set_from_lu (self):
        P, L, U = lu (self.A.T)
        m, n = self.A.shape
        l = min(m, n)
        A_T_approx = L[:l][:,:l] @ U[:l][:,:l]
        print(f"P:\n{P}\nL:\n{L}\nU:\n{U}\n") #se n > m tutto torna 
        print(f"approx:\n{A_T_approx}\nA.T:\n{self.A.T}")

        self.B.fill(False)
        self.N.fill(True)

        for row in A_T_approx:
            for i, a_row in enumerate(self.A.T):
                if (np.allclose(row, a_row, atol=self.tol)):
                    self.B[i] = True

        print(f"B:\n{self.B}\nN:\n{~self.B}")
        print(f"A[B]:\n{self.A[:, self.B]}\nA[N]:\n{self.A[:, ~self.B]}")

        """
        Z = np.block([
            [-np.linalg.pinv(self.A[:, self.B]) @ self.A[:, ~self.B]],
            [np.eye(max(m,n) - l)]
        ])

        tmp_H = Z.T @ self.H @ Z

        R = cholesky(tmp_H)

        print(f"Z:\n{Z}\nZ.THZ:\n{tmp_H}\nR:\n{R}")

        mask = np.full(max(n,m)-l, False)
        for i in range(R.shape[0]):
            if (R[i, i] is not 0):
                mask[i] = True

        #self.B[~self.B] = mask

        """ #stuff che potrebbe essere utile in futuro
        self.N = ~self.B
        print(f"B:\n{self.B}\nN:\n{self.N}")

    def set_initial_solution_from_basis (self): #section 5.2 del paper
        """! Function that calculate the initial solution for the primal problem
        
        """
        #qui pongo x[N] al lower bound se l != -inf, e all'upper altrimenti. Se anche l'upper è +inf allora 0, che è un casino non implementato
        self.x[self.N] = np.where(self.l[self.N] == -np.inf, self.u[self.N]+self.q_u[self.N], self.l[self.N]-self.q_l[self.N])
        self.x[self.N] = np.where(self.x[self.N] == np.inf, 0, self.x[self.N]) #forse da fare i tmp bounds invece che 0
        self.z_l[self.B] = -self.r_l[self.B] #questi sono ok perché con B x non è a nessuno dei due vincoli, quindi entrambe le z sono a 0 
        self.z_u[self.B] = -self.r_u[self.B]

#        self.logger.info(f"x[N]:\n{self.x[self.N]}\nq[N]:\n{self.q[self.N]}")
        B_size = np.sum(self.B)
    
        K_I = np.block([
            [self.H[self.B, :][:, self.B], self.A[:, self.B].T ],
            [self.A[:, self.B],           -self.M]
        ])
        
        tmp_b = np.concatenate(
            (self.H[self.B, :][:, self.N] @ self.x[self.N] - self.c[self.B] - self.z_l[self.B] - self.z_u[self.B],
             self.A[:, self.N] @ self.x[self.N] + self.b),
            axis = 0
        )
        
        sol = solve_plin(K_I, tmp_b).reshape((B_size + self.y.size,))

#        self.logger.info(f"K_i @ x = b:\n{K_I}\n{sol}\n{tmp_b}")
#        cond = np.allclose (K_I @ sol, tmp_b, atol=self.tol)
#        self.logger.info(f"cond:\n{K_I @ sol}\n{cond}")
        
        self.x[self.B], self.y = sol[:B_size], -sol[B_size:]

        tmp_z = (self.H[self.B, :][:, self.N].T @ self.x[self.B] -
                 self.H[self.N, :][:, self.N]   @ self.x[self.N] +
                 self.c[self.N] -
                 self.A[:, self.N].T @ self.y) #ora, qui questa z è z_u se x[N] è al limite superiore, z_l se x[N] è al limite inferiore
        self.z_u[self.N] = np.where(np.allclose(self.x[self.N], self.u[self.N]+self.q_u[self.N], atol=self.tol), tmp_z, 0)
        self.z_l[self.N] = np.where(np.allclose(self.x[self.N], self.l[self.N]-self.q_l[self.N], atol=self.tol), tmp_z, 0)
        #sopra pongo le z ad un valore di tmp_z se x è al corrispettivo bound, altrimenti 0

        max_q_l = np.where(self.l[self.B]-self.x[self.B] > 0, self.l[self.B]-self.x[self.B], 0)
        max_q_u = np.where(self.x[self.B]-self.u[self.B] > 0, self.x[self.B]-self.u[self.B], 0)
        max_r_l = np.where(-self.z_l[self.N] > 0, -self.z_l[self.N], 0)
        max_r_u = np.where(-self.z_u[self.N] > 0, -self.z_u[self.N], 0)
        self.q_l[self.B] = np.where(self.q_l[self.B] > max_q_l, self.q_l[self.B], max_q_l)
        self.q_u[self.B] = np.where(self.q_u[self.B] > max_q_u, self.q_u[self.B], max_q_u)
        self.r_l[self.N] = np.where(self.r_l[self.N] > max_r_l, self.r_l[self.N], max_r_l)
        self.r_u[self.N] = np.where(self.r_u[self.N] > max_r_u, self.r_u[self.N], max_r_u)
        
        self.logger.info(f"the generated solutions from the B and N sets are:\nx:\n{self.x}\ny:\n{self.y}\nz_l:\n{self.z_l}\nz_u:\n{self.z_u}")
        self.logger.info(f"the new constrains vectors are:\nq_l:\n{self.q_l}\nq_u:\n{self.q_u}\nr_l:\n{self.r_l}\nr_u:\n{self.r_u}")

    def reset_deltas (self):
        """! Function that reset the deltas values to 0
       
        """
        self.dx.fill(0)
        self.dy.fill(0)
        self.dz.fill(0)
        self.dz_l.fill(0)
        self.dz_u.fill(0)

    def get_solution(self):
        """! Function that return the solution of the Quadratic Problem

        @return The result of the problem [ cx + 0.5x.THx + 0.5y.TMy ] with the current solution that satisfy the constrains [Ax + My = b ] and [x >= 0]
        """
        constraint_AMb = self.A @ self.x + self.M @ self.y - self.b
        assert np.allclose(norm_2(constraint_AMb), 0, atol=self.tol), constraint_AMb
        constraint_x = (self.x >= self.l-self.tol) and (self.x <= self.u+self.tol)
        assert np.allclose(constraint_x, True, atol=self.tol), constraint_x
        sol = self.c @ self.x + 0.5* self.x.T @ self.H @ self.x + 0.5 * self.y.T @ self.M @ self.y
        self.logger.info(f"The solution of the system is: {sol}")
        return sol

    def test_primal_feasible(self, relaxed = False):
        """! Function that check if the current solution satisfy the condition to be a feasible solution for the Primal problem

        @return True if every condition is satisfied
        """
        condition_1 = self.A @ self.x + self.M @ self.y - self.b
        self.logger.info(f"Ax + My - b: {condition_1}")
        assert np.allclose(norm_2(condition_1), 0, atol=self.tol), condition_1
        condition_2 = (self.H[self.B, :][:, self.B] @ self.x[self.B] +
                       self.H[self.B, :][:, self.N] @ self.x[self.N] +
                       self.c[self.B] -
                       self.A[:, self.B].T @ self.y - self.z[self.B])
        self.logger.info(f"H[bb]x[b]: + H[bn]x[n]:\n + c[b] - A[b].Ty - z[b]: {condition_2}") 
        assert np.allclose(norm_2(condition_2), 0, atol=self.tol), condition_2
        condition_3 = (self.H[self.B, :][:, self.N].T @ self.x[self.B] + 
                       self.H[self.N, :][:, self.N]   @ self.x[self.N] + self.c[self.N] - 
                       self.A[:, self.N].T            @ self.y         - self.z_l[self.N] - self.z_u[self.N])
        self.logger.info(f"H[bn].Tx + H[nn]x + c[n] + A[n].Ty - z[n]: {condition_3}") 
        assert np.allclose(norm_2(condition_3), 0, atol=self.tol), condition_3

        #lower bound conditions, 
        condition_4_l = self.z_l[self.B] + self.r_l[self.B]
        self.logger.info(f"z_l[b] + r_l[b]: {condition_4_l}")
        if (not relaxed):
            assert np.allclose(condition_4_l, 0, atol=self.tol), condition_4 #normal condition
        else:
            assert np.all(condition_4_l <= 0.+self.tol), condition_4 #relaxed condition
        condition_5_l = np.allclose(self.x[self.N] + self.q_l[self.N], self.l[self.N], atol=self.tol)
        self.logger.info(f"x[n] + q_l[n]: {condition_5_l}")
        condition_6_l = (self.x[self.B] + self.q_l[self.B] >= self.l[self.B]-self.tol)

        #upper bound conditions
        condition_4_u = self.z_u[self.B] + self.r_u[self.B]
        self.logger.info(f"z_u[b] + r_u[b]: {condition_4_u}")
        if (not relaxed):
            assert np.allclose(condition_4_u, 0., atol=self.tol), condition_4 #normal condition
        else:
            assert np.all(condition_4_u <= 0.+self.tol), condition_4 #relaxed condition
        condition_5_u = np.allclose (self.x[self.N] + self.q_u[self.N], self.u[self.N], atol=self.tol)
        self.logger.info(f"x[n] + q_u[n]: {condition_5_u}")
        condition_6_u = (self.x[self.B] + self.q_u[self.B] <= self.u[self.B]+self.tol)
            
        condition_5 = condition_5_l or condition_5_u
        condition_6 = condition_6_l & condition_6_u
        assert np.allclose(condition_5, True, atol=self.tol), condition_5
        self.logger.info(f"l <= x[b] + q[b] <= u: {condition_6}") 
        assert np.allclose(condition_6, True, atol=self.tol), condition_6
        return True        
    
    def test_dual_feasible(self, relaxed=False):
        """! Function that check if the current solution satisfy the condition to be a feasible solution for the Dual problem
        
        @return True if every condition is satisfied
        """
        condition_1 = self.H @ self.x + self.c - self.A.T @ self.y - self.z_l - self.z_u
        self.logger.info(f"Hx + c - A.Ty - z: {condition_1}") 
        assert np.allclose(norm_2(condition_1), 0, atol=self.tol), condition_1
        condition_2 = self.A @ self.x + self.M @ self.y - self.b
        self.logger.info(f"Ax + My - b: {condition_2}") 
        assert np.allclose(norm_2(condition_2), 0, atol=self.tol), condition_2
        
        condition_3_l = self.x[self.N] + self.q_l[self.N]
        self.logger.info(f"x[n] + q_l[n]: {condition_3_l}")
        if (not relaxed):
            condition_3_l = np.allclose(condition_3_l, self.l[self.N], atol=self.tol) #normal condition
        else:
            condition_3_l = np.all(condition_3_l <= self.l[self.N]+self.tol) #relaxed condition
        condition_4_l = self.z_l[self.B] + self.r_l[self.B]
        self.logger.info(f"z_l[b] + r_l[b]: {condition_4_l}") 
        assert np.allclose(norm_2(condition_4_l), 0, atol=self.tol), condition_4_l
        condition_5_l = (self.z_l[self.N] + self.r_l[self.N] >= 0.-self.tol)
        self.logger.info(f"z_l[n] + r_l[n] >= 0: {condition_5_l}")
        assert np.allclose(condition_5_l, True, atol=self.tol), condition_5_l
            
        condition_3_u = self.x[self.N] + self.q_u[self.N]
        self.logger.info(f"x[n] + q_u[n]: {condition_3_u}")
        if (not relaxed):
            condition_3_u = np.allclose(condition_3_u, self.u[self.N], atol=self.tol) #normal condition
        else:
            condition_3_u = np.all(condition_3_u <= self.u[self.N]+self.tol) #relaxed condition
        condition_4_u = self.z_u[self.B] + self.r_u[self.B]
        self.logger.info(f"z_u[b] + r_u[b]: {condition_4_u}")
        assert np.allclose(norm_2(condition_4_u), 0, atol=self.tol), condition_4_u
        condition_5_u = (self.z_u[self.N] + self.r_u[self.N] >= 0.-self.tol)
        self.logger.info(f"z_u[n] + r_u[n] >= 0: {condition_5_u}")
        assert np.allclose(condition_5_u, True, atol=self.tol), condition_5_u

        condition_3 = condition_3_l or condition_3_u
        assert np.allclose(condition_3, True, atol=self.tol), condition_3

        return True

    def general_active_set (self):
        """! Function that do the primal or dual algorith given the feasibility of the initial solution
        
        @return True if he found and executed a feasible algorith
        @return False if the initial solution isn't feasible for any algorithm
        """
        try:
            self.test_primal_feasible()
            self.logger.info(f"The current set of variables is feasible for a Primal algorithm")
            self.primal_active_set()
            return True
        except AssertionError as err:
            self.logger.error(f"The current set of variables is not feasible for a Primal algorithm")
        try:
            self.test_dual_feasible()
            self.logger.info(f"The current set of variables is feasible for a Dual algorithm")
            self.dual_active_set()
            return True
        except AssertionError as err:
            self.logger.error(f"The current set of variables is not feasible for a Dual algorithm")

        self.logger.error(f"The current set of variables is not feasible for any algorithm")
        return False

    def solve (self, B=None, N=None):
        """! Function that do the primal first strategy or dual first strategy given the feasibility of the initial solution

        @return True if he found and executed a feasible algorith
        @return False if the initial solution isn't feasible for any algorithm
        """
        self.logger.info(f"-"*20)
        self.logger.info(f"Started generic Solver")
        self.logger.info(f"Initializing the sets and the variables")
        if (B is not None and N is not None):
            self.set_initial_active_set(B, N)
        else:
            self.set_initial_active_set_from_lu()
        self.set_initial_solution_from_basis()
        old_q_l = self.q_l
        old_q_u = self.q_u
        old_r_l = self.r_l
        old_r_u = self.r_u
        try:
            self.r_l.fill(0)
            self.r_l.fill(0)
            self.test_primal_feasible()
            self.logger.info(f"The current set of variables is feasible for a Primal First algorithm")
            return self.primal_first_strategy()
        except AssertionError as err:
            self.r_l = old_r_l
            self.r_u = old_r_l
            self.logger.error(f"The current set of variables is not feasible for a Primal First algorithm: {err}")
        try:
            self.q_l.fill(0)
            self.q_u.fill(0)
            self.test_dual_feasible()
            self.logger.info(f"The current set of variables is feasible for a Dual First algorithm")
            return self.dual_first_strategy()
        except AssertionError as err:
            self.q_l = old_q_l
            self.q_u = old_q_u
            self.logger.error(f"The current set of variables is not feasible for a Dual First algorithm: {err}")

        self.logger.error(f"The current set of variables is not feasible for any algorithm")
        return None

    def primal_first_strategy(self):
        """! Function that do the Primal Shift Strategy
        @param B the basis vector
        @param N the basis vector opposite
        @return self.get_solution on the optimal solutions found
        """
        self.logger.info(f"-"*20)
        self.logger.info(f"Starting the primal first Strategy for solving the original problem with shifts")
#        self.logger.info(f"Initializing the sets and the variables")
#        self.set_initial_active_set(B, N)
#        self.set_initial_active_set_from_factorization()
#        self.set_initial_solution_from_basis()
        
        self.logger.info(f"Resetting the r vector and starting the primal problem")
        self.r_l.fill(0)
        self.r_u.fill(0)
        self.test_primal_feasible()
        self.primal_active_set()

        self.logger.info(f"Resetting the q vector and starting the dual problem - hiyo")
        self.q_l.fill(0)
        self.q_u.fill(0)
        self.test_dual_feasible(relaxed=True)
        self.dual_active_set()

        self.logger.info(f"The Primal Shift Strategy ended with success")
        return self.get_solution()

    def dual_first_strategy(self):
        """! FUnction that does the Dual First Strategy

        @param B the basis vector
        @param N the basis vector opposite

        @return self.get_solution of the optimal solution found
        """

        self.logger.info(f"-"*20)
        self.logger.info(f"Starting the dual first Strategy for solving the original problem with shifts")
#        self.logger.info(f"Initializing the sets and the variables")
#        self.set_initial_active_set(B, N)
#        self.set_initial_active_set_from_factorization()
#        self.set_initial_solution_from_basis()

        self.logger.info(f"Resetting the q vector and starting the dual problem")
        self.q_l.fill(0)
        self.q_u.fill(0)
        self.test_dual_feasible()
        self.dual_active_set()

        self.logger.info(f"Resetting the r vector and starting the primal problem")
        self.r_l.fill(0)
        self.r_u.fill(0)
        self.test_primal_feasible(relaxed=True)
        self.primal_active_set()

        self.logger.info(f"The Dual Shift Strategy ended with success")
        return self.get_solution()
    
    def primal_active_set(self):
        """! Function that start the active set Primal Algorithm

        @return x vector R(n) with the optimal solution
        @return y vector R(m) with the optimal solution
        @return z vector R(n) with the optimal solution
        """
        self.logger.info("-"*20)
        self.logger.info(f"Starting the resolution of the Primal Problem of the Active Sets")
        
        # --------------- Inizio loop principale
        while True:
            l_list = np.argwhere(((self.z_l + self.r_l) < 0.-self.tol) | ((self.z_u + self.r_u) < 0.-self.tol)) #questo da un qualsiasi indice che viola i vincoli
            self.logger.info(f"The indexes that violate the constrains are: {l_list.flatten()}")
            if l_list.size == 0:
                self.logger.info(f"The primal algorith just terminated its course. The solutions are as follows:")
                self.logger.info(f"x:\n{self.x}")
                self.logger.info(f"y:\n{self.y}")
                self.logger.info(f"z:\n{self.z}")
                return (
                    self.x,
                    self.y,
                    self.z,
                )  # non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            #l = l_list[0]
            l = l_list[np.min(np.argmin(self.z_l[l_list] + self.r_l[l_list]), np.argmin(self.z_u[l_list] + self.r_u[l_list]))]
            
            if (self.N[l] == True):
                self.N[l] = False  # prendo il primo elemento di l e lo levo da N.
                self.primal_base(l)
                self.reset_deltas()
            else:
                self.B[l] = False

            while (self.z_l[l] + self.r_l[l]) < 0.-self.tol or (self.z_u[l] + self.r_u[l]) < 0.-self.tol:
#                input("premi per continuare...")
                self.primal_intermediate(l)
                self.reset_deltas()
            self.B[l] = True

    def primal_base(self, l):
        """! Function that do the base iteration of the Primal problem

        @param l the index that violate the constrain
        """
        self.logger.info("-"*20)
        self.logger.info(f"Base iteration of the Primal Problem, with the index: {l}")
        
        self.dx[l] = 1.
        B_size = np.sum((self.B))
        
        self.logger.info(f"Hbb:\n{self.H[self.B, :][:, self.B]}")
        self.logger.info(f"Ab:\n{self.A[:, self.B]}")
        self.logger.info(f"M:\n{self.M}")
        K_I = np.block ([
            [self.H[self.B, :][:, self.B], self.A[:, self.B].T],
            [self.A[:, self.B],           -self.M             ],
        ])
        self.logger.info(f"K_I:\n{K_I}")
        
        self.logger.info(f"Hb:\n{self.H[self.B][:, l]}")
        self.logger.info(f"Al:\n{self.A[:, l]}")
        tmp_b = -np.concatenate(
            (self.H[self.B][:, l],
             self.A[:, l]),
            axis=0
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = solve_plin(K_I, tmp_b).reshape((B_size + self.y.size,)) #da cambiare
        self.logger.info(f"sol:\n{tmp_sol}")
        
        self.dx[self.B], self.dy[:] = tmp_sol[:B_size], -tmp_sol[B_size:]
        self.logger.info(f"delta x:\n{self.dx}")
        self.logger.info(f"delta y:\n{self.dy}")

        tmp_dz = (
              self.H[self.N, l]           * self.dx[l] #qui * va bene perché delta_x_l è uno scalare
            + self.H[self.B][:, self.N].T @ self.dx[self.B] #qui usiamo matmul perché è la moltiplicazione di una matrice #Nx#B per un vettore #Bx1
            - self.A[:, self.N].T         @ self.dy #come sopra, #Nxm per mx1
        )
        tmp_dz_l = (
              self.H[l, l]           * self.dx[l] #scalare
            + self.H[self.B][:, l].T @ self.dx[self.B] # qui è un vettore 1x#B per #B
            - self.A[:, l].T         @ self.dy # qui è 1xm per mx1
        )
        if (self.z_l[l] + self.r_l[l] < 0.-self.tol):
            self.dz_l[self.N] = tmp_dz
            self.dz_l[l] = tmp_dz_l
            self.logger.info(f"delta z_l:\n{self.dz_l}")

            alpha_opt = math.inf if np.allclose(self.dz_l[l], 0, atol=self.tol) else -(self.z_l[l] + self.r_l[l]) / self.dz_l[l]
    
        if (self.z_u[l] + self.r_u[l] < 0.-self.tol): #questi due if, in teoria, sono uno l'opposto dell'altro. o vale uno o vale l'altro.
            self.dz_u[self.N] = tmp_dz
            self.dz_u[l] = tmp_dz_l
            self.logger.info(f"delta z_u:\n{self.dz_u}")

            alpha_opt = math.inf if np.allclose(self.dz_u[l], 0, atol=self.tol) else -(self.z_u[l] + self.r_u[l]) / self.dz_u[l]

        min_mask = self.B & (self.dx != 0)
        to_min = np.where(self.dx < 0, (self.x - self.l + self.q_l), (self.x - self.u - self.q_u))
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
        self.logger.info(f"to_min:\n{to_min[self.B]}\n")
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt: {alpha_opt}; max: {alpha_max});")
        
        if np.isinf(alpha):
            self.logger.exception(f"Primal is Unbounded (Dual is unfeasible")
            raise Exception("Primal is Unboundend (Dual is unfeasible)")  # il problema è impraticabile

        #if np.isclose(alpha, 0, atol=self.tol):
        #    self.logger.exception(f"Step size is zero")
        #    raise Exception("Step size is zero")
            
        self.x[l]        += alpha * self.dx[l]
        self.x[self.B]   += alpha * self.dx[self.B]
        self.y           += alpha * self.dy
        self.z_l[l]      += alpha * self.dz_l[l]
        self.z_l[self.N] += alpha * self.dz_l[self.N]
        self.z_u[l]      += alpha * self.dz_u[l]
        self.z_u[self.N] += alpha * self.dz_u[self.N]
        
        if self.z_l[l] + self.r_l[l] < 0.-self.tol or self.z_u[l] + self.r_u[l] < 0.-self.tol: #effettivamente anche alpha == alpha_max ha senso
            self.logger.info(f"k: {k}")
            self.B[k] = False
            self.N[k] = True
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z_l:\n{self.z_l}")
        self.logger.info(f"z_u:\n{self.z_u}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

    def primal_intermediate(self, l):
        """! Function that do the intermediate step of the Primal problem

        @param l the index that violate the constrain
        """
        self.logger.info(f"-"*20)
        self.logger.info(f"Intermediate step of the Primal problem, done with the index: {l}")

        B_size = np.sum((self.B))

        self.logger.info(f"Hll:\n{self.H[l, l]}")
        self.logger.info(f"HBl:\n{self.H[self.B][:, l]}")
        self.logger.info(f"Hbb:\n{self.H[self.B][:, self.B]}")
        self.logger.info(f"Al:\n{self.A[:, l]}")
        self.logger.info(f"Ab:\n{self.A[:, self.B]}")
        self.logger.info(f"M:\n{self.M}")
        K_I = np.block([
            [self.H[l, l],         self.H[self.B, l].T,       self.A[:, l].T     ],
            [self.H[self.B][:, l], self.H[self.B][:, self.B], self.A[:, self.B].T],
            [self.A[:, l],         self.A[:, self.B],         -self.M]
        ])
        self.logger.info(f"K_I:\n{K_I}")
        
        tmp_b = np.concatenate((
            np.ones(1),
            np.zeros(B_size + self.y.size))
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = solve_plin(K_I, tmp_b).reshape((1+B_size+self.y.size,))
        self.logger.info(f"sol:\n{tmp_sol}")
        # robo da risolvere
        self.dx[l], self.dx[self.B], self.dy[:] = tmp_sol[0], tmp_sol[1: B_size +1], -tmp_sol[B_size + 1:]
        self.logger.info(f"delta x:\n{self.dx}")
        self.logger.info(f"delta y:\n{self.dy}")

        tmp_dz = (
              self.H[self.N, l]           * self.dx[l] #scalare 
            + self.H[self.B][:, self.N].T @ self.dx[self.B] #moltiplicazione #Nx#B per #Bx1
            - self.A[:, self.N].T         @ self.dy[:] # moltiplicazione #Nxm per #mx1
        )
        self.logger.info(f"delta z\n{self.dz}")

        if (self.z_l[l] + self.r_l[l] < 0.-self.tol):
            self.dz_l[self.N] = tmp_dz
            self.dz_l[l] = 1
            
            alpha_opt = -(self.z_l[l] + self.r_l[l])
            
        if (self.z_u[l] + self.r_u[l] < 0.-self.tol):
            self.dz_u[self.N] = tmp_dz
            self.dz_u[l] = 1
            
            alpha_opt = -(self.z_l[l] + self.r_l[l])
        
        min_mask = self.B & (self.dx != 0)
        to_min = np.where(self.dx < 0, (self.x - self.l + self.q_l), (self.x - self.u - self.q_u))
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
        self.logger.info(f"to_min:\n{to_min[self.B]}")
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt = {alpha_opt}; max = {alpha_max});")
        
        if np.isclose(alpha, 0, atol=self.tol):
            self.logger.exception(f"Step size is zero")
            raise Exception("Step size is zero")
        
        self.x[l]        += alpha * self.dx[l]
        self.x[self.B]   += alpha * self.dx[self.B]
        self.y           += alpha * self.dy
        self.z_l[l]      += alpha * self.dz_l[l]
        self.z_l[self.N] += alpha * self.dz_l[self.N]
        self.z_u[l]      += alpha * self.dz_u[l]
        self.z_u[self.N] += alpha * self.dz_u[self.N]
        
        if self.z_l[l] + self.r_l[l] < 0.-self.tol or self.z_u[l] + self.r_u[l] < 0.-self.tol:
            self.logger.info(f"k: {k}")
            self.B[k] = False
            self.N[k] = True
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z_l:\n{self.z_l}")
        self.logger.info(f"z_u:\n{self.z_u}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

    def dual_active_set(self):
        """! Function that start the active set Dual Algorithm

        @return x vector R(n) with the optimal solution
        @return y vector R(m) with the optimal solution
        @return z vector R(n) with the optimal solution
        """
        self.logger.info("-"*20)
        self.logger.info(f"Starting the resolution of the Dual Problem of the Active Sets")
        
        # --------------- Inizio loop principale
        while True:
            l_list = np.argwhere((self.x + self.q) < 0.-self.tol)
            self.logger.info(f"The indexes that violate the constrains are: {l_list.flatten()}")
            if l_list.size == 0:
                self.logger.info(f"The dual algorith just terminated its course. The solutions are as follows:")
                self.logger.info(f"x:\n{self.x}")
                self.logger.info(f"y:\n{self.y}")
                self.logger.info(f"z:\n{self.z}")
                return (
                    self.x,
                    self.y,
                    self.z,
                )  # non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            #l = l_list[0]
            l = l_list[np.argmin(self.x[l_list] + self.q[l_list])]
            
            if (self.B[l] == True):
                self.B[l] = False  # prendo il primo elemento di l e lo levo da N.
                self.dual_base(l)
                self.reset_deltas()
            else:
                self.N[l] = False
            
            while (self.x[l] + self.q[l]) < 0.-self.tol:
#                input("premi per continuare...")
                self.dual_intermediate(l)
                self.reset_deltas()
            self.N[l] = True

    def dual_base(self, l):
        """! Function that do the base step of the Dual problem

        @param l the index that violate the constrain
        """
        self.logger.info(f"-"*20)
        self.logger.info(f"Base step of the Dual problem, done with the index: {l}")
        
        self.dz[l] = 1
        B_size = np.sum((self.B))

        self.logger.info(f"Hll:\n{self.H[l, l]}")
        self.logger.info(f"HBl:\n{self.H[self.B][:, l]}")
        self.logger.info(f"Hbb:\n{self.H[self.B][:, self.B]}")
        self.logger.info(f"Al:\n{self.A[:, l]}")
        self.logger.info(f"Ab:\n{self.A[:, self.B]}")
        self.logger.info(f"M:\n{self.M}")
        K_I = np.block([
            [self.H[l, l],         self.H[self.B, l].T,       self.A[:, l].T     ],
            [self.H[self.B][:, l], self.H[self.B][:, self.B], self.A[:, self.B].T],
            [self.A[:, l],         self.A[:, self.B],         -self.M]
        ])
        self.logger.info(f"K_I:\n{K_I}")
            
        tmp_b = np.concatenate((
            np.ones(1),
            np.zeros(B_size + self.y.size))
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = solve_plin(K_I, tmp_b).reshape((1+B_size+self.y.size,))
        self.logger.info(f"sol:\n{tmp_sol}")
        # robo da risolvere
        self.dx[l], self.dx[self.B], self.dy[:] = tmp_sol[0], tmp_sol[1: B_size +1], -tmp_sol[B_size + 1:]
        self.logger.info(f"delta x:\n{self.dx}")
        self.logger.info(f"delta y:\n{self.dy}")

        self.dz[self.N] = (
              self.H[self.N, l]           * self.dx[l] #scalare 
            + self.H[self.B][:, self.N].T @ self.dx[self.B] #moltiplicazione #Nx#B per #Bx1
            - self.A[:, self.N].T         @ self.dy[:] # moltiplicazione #Nxm per #mx1
        )
        self.logger.info(f"delta z\n{self.dz}")
        
        alpha_opt = np.inf if np.allclose(self.dx[l], 0, atol=self.tol) else -(self.x[l] + self.q[l])/self.dx[l]
        min_mask = ( self.dz < 0 )
        to_min = self.r + self.z
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dz[min_mask]
        print(f"min_mask: {min_mask}")
        print(f"z: {self.x[min_mask]} r: {self.q[min_mask]}")
        self.logger.info(f"to_min:\n{to_min}")
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt = {alpha_opt}; max = {alpha_max});")

        if np.isinf(alpha):
            self.logger.exception(f"Dual is Unbounded (Primal is unfeasible)")
            raise Exception("Dual is Unboundend (Primal is unfeasible)")
        
#        if np.isclose(alpha, 0, atol=self.tol):
#            self.logger.exception(f"Step size is zero")
#            raise Exception("Step size is zero")
        
        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.x[l] + self.q[l] < 0.-self.tol:
            self.logger.info(f"k: {k}")
            self.B[k] = True
            self.N[k] = False
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z:\n{self.z}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

    def dual_intermediate(self, l):
        """! Function that do the intermediate iteration of the Dual problem

        @param l the index that violate the constrain
        """
        self.logger.info("-"*20)
        self.logger.info(f"Intermadiate iteration of the Dual Problem, with the index: {l}")
        
        self.dx[l] = 1.
        B_size = np.sum((self.B))
        
        self.logger.info(f"Hbb:\n{self.H[self.B, :][:, self.B]}")
        self.logger.info(f"Ab:\n{self.A[:, self.B]}")
        self.logger.info(f"M:\n{self.M}")
        K_I = np.block ([
            [self.H[self.B, :][:, self.B], self.A[:, self.B].T],
            [self.A[:, self.B],           -self.M             ],
        ])
        self.logger.info(f"K_I:\n{K_I}")
        
        self.logger.info(f"Hb:\n{self.H[self.B][:, l]}")
        self.logger.info(f"Al:\n{self.A[:, l]}")
        tmp_b = -np.concatenate(
            (self.H[self.B][:, l],
             self.A[:, l]),
            axis=0
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = solve_plin(K_I, tmp_b).reshape((B_size + self.y.size,)) #da cambiare
        self.logger.info(f"sol:\n{tmp_sol}")
        
        self.dx[self.B], self.dy[:] = tmp_sol[:B_size], -tmp_sol[B_size:]
        self.logger.info(f"delta x:\n{self.dx}")
        self.logger.info(f"delta y:\n{self.dy}")

        self.dz[self.N] = (
              self.H[self.N, l]           * self.dx[l] #qui * va bene perché delta_x_l è uno scalare
            + self.H[self.B][:, self.N].T @ self.dx[self.B] #qui usiamo matmul perché è la moltiplicazione di una matrice #Nx#B per un vettore #Bx1
            - self.A[:, self.N].T         @ self.dy #come sopra, #Nxm per mx1
        )
        self.dz[l] = (
              self.H[l, l]           * self.dx[l] #scalare
            + self.H[self.B][:, l].T @ self.dx[self.B] # qui è un vettore 1x#B per #B
            - self.A[:, l].T         @ self.dy # qui è 1xm per mx1
        )
        self.logger.info(f"delta z:\n{self.dz}")

        alpha_opt = -(self.x[l] + self.q[l])

        min_mask = ( self.dz < 0 )
        to_min = self.z + self.r
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dz[min_mask]
        print(f"min_mask: {min_mask}")
        print(f"x: {self.x[min_mask]} q: {self.q[min_mask]}")
        self.logger.info(f"to_min:\n{to_min}")
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt: {alpha_opt}; max: {alpha_max});")
        
        if np.isinf(alpha):
            self.logger.exception(f"Dual is Unbounded (Primal is unfeasible")
            raise Exception("Dual is Unboundend (Primal is unfeasible)")  # il problema è impraticabile
            
        if np.isclose(alpha, 0, atol=self.tol):
            self.logger.exception(f"Step size is zero")
            raise Exception("Step size is zero")

        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.x[l] + self.q[l] < 0.-self.tol: #effettivamente anche alpha == alpha_max ha senso
            self.logger.info(f"k: {k}")
            self.B[k] = True
            self.N[k] = False
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z:\n{self.z}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing of the active set method')
    parser.add_argument('--seed',
                        help='random seed of the operation')
    parser.set_defaults(seed=2021)
    args = parser.parse_args()
    
    n, m = 100, 50
    np.random.seed(int(args.seed))

    A = 2*(np.random.rand(m, n)-np.random.rand(m, n))
#    M = 100*np.eye(m) + np.random.rand(m, m)
#    M = M @ M.T
    M = np.zeros((m, m))
    H = 100*np.eye(n) + np.random.rand(n, n)
    H = H @ H.T
    
    b = np.random.rand(m)
    c = np.random.rand(n)
    qp = quadratic_problem (A, b, c, H, M, verbose = True)

    B = (np.random.rand(n) - np.random.rand(n)) > 0
    N = ~B
#    qp.set_initial_active_set_from_lu()
    print(qp.solve())
    pass
