#!bin/usr/python
# coding=utf-8

import numpy as np
import math
import logging
import sys

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

    def __init__(self, A, b, c, H, M, q=None, r=None, tol=1e-8, verbose=False):
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

        self.q = self.__check_shape(q, dim=(n,), varname="q") if q else np.zeros(n)
        self.r = self.__check_shape(r, dim=(n,), varname="r") if r else np.zeros(n)

        # init solutions
        self.x = np.zeros(n)
        self.y = np.zeros(m)
        self.z = np.zeros(n)
        
        # init deltas
        self.dx = np.zeros(n)
        self.dy = np.zeros(m)
        self.dz = np.zeros(n)

        # init B, N (temporary)
        self.B = np.full(n, True)
        self.N = np.full(n, False)

    def set_initial_solution(self, x, y, z):
        """! Function that set the initial solution of the problem
        
        @param x initial values of the x vector R(n)
        @param y initial values of the y vector R(m)
        @param z initial values of the z vector R(n)
        """
        self.x[:] = self.__check_shape(x, dim=(n,), varname="x")
        self.y[:] = self.__check_shape(y, dim=(m,), varname="y")
        self.z[:] = self.__check_shape(z, dim=(n,), varname="z")
        self.logger.info(f"Successfully set the initial solutions:\nx:\n{self.x}\ny:\n{self.y}\nz:\n{self.z}")
        
    def set_initial_active_set(self, B, N):
        """! Function that set the initial active set of the problem
        
        @param B initial values of B boolean vector R(n)
        @param N initial values of N boolean vector R(n)
        """
        assert all(np.logical_xor(B, N)), "Sets are not valid. |Union| should be n and |Intersection| should be 0"
        self.B[:] = self.__check_shape(B, dim=(n,), varname="B")
        self.N[:] = self.__check_shape(N, dim=(n,), varname="N")
        self.logger.info(f"Successfully set the initial active set:\nB:\n{self.B}\nN:\n{self.N}")

    def reset_deltas (self):
        self.dx.fill(0)
        self.dy.fill(0)
        self.dz.fill(0)

    def get_solution(self):
        """! Function that return the solution of the Quadratic Problem

        @return The result of the problem [ cx + 0.5x.THx + 0.5y.TMy ] with the current solution that satisfy the constrains [Ax + My = b ] and [x >= 0]
        """
        constraint_AMb = self.A @ self.x + self.M @ self.y - self.b
        assert np.allclose(constraint_AMb, 0, rtol=self.tol), constraint_AMb
        constraint_x = (self.x >= 0)
        assert np.allclose(constraint_x, True, rtol=self.tol), constraint_x
        sol = self.c @ self.x + 0.5* self.x.T @ self.H @ self.x + 0.5 * self.y.T @ self.M @ self.y
        self.logger.info(f"The solution of the system is: {sol}")
        return sol

    def test_primal_feasible(self):
        """! Function that check if the current solution satisfy the condition to be a feasible solution for the Primal problem

        @return True if every condition is satisfied
        """
        condition_1 = self.A @ self.x + self.M @ self.y - self.b
        self.logger.info(f"Ax + My - b: {condition_1}")
        assert np.allclose(condition_1, 0, rtol=self.tol), condition_1
        condition_2 = (self.H[self.B]      @ self.x + self.c[self.B] - 
                       self.A[:, self.B].T @ self.y - self.z[self.B])
        self.logger.info(f"H[b]x + c[b] - A[b].Ty - z[b]: {condition_2}") 
        assert np.allclose(condition_2, 0, rtol=self.tol), condition_2
        condition_3 = (self.H[self.B, :][:, self.N].T @ self.x[self.B] + 
                       self.H[self.N, :][:, self.N]   @ self.x[self.N] + self.c[self.N] - 
                       self.A[:, self.N].T            @ self.y         - self.z[self.N])
        self.logger.info(f"H[bn].Tx + H[nn]x + c[n] + A[n].Ty - z[n]: {condition_3}") 
        assert np.allclose(condition_3, 0, rtol=self.tol), condition_3
        condition_4 = self.z[self.B] + self.r[self.B]
        self.logger.info(f"z[b] + r[b]: {condition_4}") 
        assert np.allclose(condition_4, 0, rtol=self.tol), condition_4
        condition_5 = self.x[self.N] + self.q[self.N]
        self.logger.info(f"x[n] + q[n]: {condition_5}") 
        assert np.allclose(condition_5, 0, rtol=self.tol), condition_5
        condition_6 = (self.x[self.B] + self.q[self.B] >= 0)
        self.logger.info(f"x[b] + q[b] >= 0: {condition_6}") 
        assert np.allclose(condition_6, True, rtol=self.tol), condition_6
        return True
    
    def test_dual_feasible(self):
        """! Function that check if the current solution satisfy the condition to be a feasible solution for the Dual problem
        
        @return True if every condition is satisfied
        """
        condition_1 = self.H @ self.x + self.c - A.T @ self.y - self.z
        self.logger.info(f"Hx + c - A.Ty - z: {condition_1}") 
        assert np.allclose(condition_1, 0, rtol=self.tol), condition_1
        condition_2 = self.A @ self.x + self.M @ self.y - self.b
        self.logger.info(f"Ax + My - b: {condition_2}") 
        assert np.allclose(condition_2, 0, rtol=self.tol), condition_2
        condition_3 = self.x[self.N] + self.q[self.N]
        self.logger.info(f"x[n] + q[n]: {condition_3}") 
        assert np.allclose(condition_3, 0, rtol=self.tol), condition_3
        condition_4 = self.z[self.B] + self.r[self.B]
        self.logger.info(f"z[b] + r[b]: {condition_4}") 
        assert np.allclose(condition_4, 0, rtol=self.tol), condition_4
        condition_5 = (self.z[self.N] + self.r[self.N] >= 0)
        self.logger.info(f"z[n] + r[n] >= 0: {condition_5}") 
        assert np.allclose(condition_5, True, rtol=self.tol), condition_5
        return True

    def general_active_set (self):
        pdb = np.array([False, False])
        try:
            pdb[0] = self.test_primal_feasible()
            self.logger.info(f"The current set of variables is feasible for a Primal algorithm")
            self.primal_active_set()
            return True
        except AssertionError as err:
            self.logger.error(f"The current set of variables is not feasible for a Priaml algorithm")
        try:
            pdb[1] = self.test_dual_feasible()
            self.logger.info(f"The current set of variables is feasible for a Dual algorithm")
            self.dual_active_set()
            return True
        except AssertionError as err:
            self.logger.error(f"The current set of variables is not feasible for a Dual algorithm")
        if not any(pdb):
            self.logger.error(f"The current set of variables is not feasible for any algorithm")
            return False
    
    def primal_active_set(self):
        """! Function that start the active set Primal Algorithm

        @return x vector R(n) with the optimal solution
        @return y vector R(m) with the optimal solution
        @return z vector R(n) with the optimal solution
        """
        self.logger.info("-"*20)
        self.logger.info(f"Starting the resolution of the Primal Problem of the Active Sets")

        res = self.test_primal_feasible()
        
        # --------------- Inizio loop principale
        while True:
            l_list = np.argwhere((self.z - self.r) < 0)
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
            l = l_list[0]
            self.N[l] = False  # prendo il primo elemento di l e lo levo da N.
            self.primal_base(l)
            self.reset_deltas()
            while (self.z[l] + self.r[l]) < 0:
                input("premi per continuare...")
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
            (self.H[self.B][:, l], self.A[:, l]),
            axis=0
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = np.linalg.solve(K_I, tmp_b).reshape((B_size + self.y.size,)) #da cambiare
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

        alpha_opt = math.inf if np.allclose(self.dx[l], 0, rtol=self.tol) else -(self.z[l] + self.r[l]) / self.dz[l]

        min_mask = ( self.dx < 0 )
        to_min = self.x + self.q
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
        print(f"min_mask: {min_mask}")
        print(f"x: {self.x[min_mask]} q: {self.q[min_mask]}")
        self.logger.info(f"to_min:\n{to_min}")
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt: {alpha_opt}; max: {alpha_max});")
        
        if np.isinf(alpha):
            self.logger.exception(f"Primal is Unbounded (Dual is unfeasible")
            raise Exception("Primal is Unboundend (Dual is unfeasible)")  # il problema è impraticabile

        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.z[l] + self.r[l] < 0: #effettivamente anche alpha == alpha_max ha senso
            self.logger.info(f"k: {k}")
            self.B[k] = False
            self.N[k] = True
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z:\n{self.z}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

    def primal_intermediate(self, l):
        """! Function that do the intermediate step of the Primal problem

        @param l the index that violate the constrain
        """
        self.logger.info(f"-"*20)
        self.logger.info(f"Intermediate step of the Primal problem, done with the index: {l}")
        
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
        
        tmp_sol = np.linalg.solve(K_I, tmp_b).reshape((1+B_size+self.y.size,))
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
        
        alpha_opt = -(self.z[l] + self.r[l])
        min_mask = ( self.dx < 0 )
        to_min = self.x + self.q
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
        print(f"min_mask: {min_mask}")
        print(f"x: {self.x[min_mask]} q: {self.q[min_mask]}")
        self.logger.info(f"to_min:\n{to_min}")
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt = {alpha_opt}; max = {alpha_max});")
        
        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.z[l] + self.r[l] < 0:
            self.logger.info(f"k: {k}")
            self.B[k] = False
            self.N[k] = True
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z:\n{self.z}")
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
            l_list = np.argwhere((self.x - self.q) < 0)
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
            l = l_list[0]
            self.B[l] = False  # prendo il primo elemento di l e lo levo da N.
            self.dual_base(l)
            self.reset_deltas()
            while (self.x[l] + self.q[l]) < 0:
                input("premi per continuare...")
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
        
        tmp_sol = np.linalg.solve(K_I, tmp_b).reshape((1+B_size+self.y.size,))
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
        
        alpha_opt = np.inf if np.allclose(self.dx[l], 0, rtol=self.tol) else -(self.x[l] + self.q[l])/self.dx[l]
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
            self.logger.exception(f"Dual is Unbounded (Primal is unfeasible")
            raise Exception("Dual is Unboundend (Primal is unfeasible)")
        
        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.x[l] + self.q[l] < 0:
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
            (self.H[self.B][:, l], self.A[:, l]),
            axis=0
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = np.linalg.solve(K_I, tmp_b).reshape((B_size + self.y.size,)) #da cambiare
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

        min_mask = ( self.dx < 0 )
        to_min = self.z + self.r
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
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

        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.x[l] + self.q[l] < 0: #effettivamente anche alpha == alpha_max ha senso
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
    n, m = 4, 6

    A = np.eye(m, n)
    A[2, 1] = 1
    M = np.eye(m, m)
    H = np.eye(n, n)
    
    b = np.array([ 1., 1., 2., 2., 0., 0.])
    c = np.array([-1., 3., 1., 1.])
    qp = quadratic_problem (A, b, c, H, M, verbose = True)

    x = np.array([ 1., 0., 0., 0.])
    y = np.array([ 0., 1., 2., 2., 0., 0.])
    z = np.array([ 0., 0.,-1.,-1.])
    B = np.array([True, True, False, False])
    N = ~B
    
    qp.set_initial_solution(x, y, z)
    qp.set_initial_active_set(B, N)
    #qp.primal_active_set()
    #qp.dual_active_set()
    qp.general_active_set()
    print(qp.get_solution())
    pass
