#!bin/usr/python3
# coding=utf-8

import numpy as np
from scipy.linalg import lu, solve, ldl, cholesky, lstsq
from scipy.sparse.linalg import splu, spsolve
import math
import logging
import sys
import argparse
import pdb

def solve_plin (A, b):
#    A_i = np.linalg.pinv (A)
#    P, L, U = lu (A)
#    c = np.linalg.pinv(L) @ b
#    x = np.linalg.pinv(U) @ c
#    print(f"{x}\n{np.linalg.inv(P)@x}")
    return spsolve(A, b)

def norm_2 (exp):
    norm = np.linalg.norm(exp, 2)
    return pow(norm, 2)

class quadratic_problem ():
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
        @param l vector R(n), default to np.zeros(n)
        @param u vector R(n), default to np.inf
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
        self.a_size = n

        # if (np.allclose(M, 0, atol=self.tol)):
        #     self.A = np.block([self.A, -np.eye(m)])
        #     c = np.concatenate((c, np.zeros(m)))
        #     l = np.concatenate((l, b))
        #     u = np.concatenate((u, b))
        #     H = np.block([[H, np.zeros((n,m))],
        #                   [np.zeros((m,n)), np.zeros((m,m))]])
        #     b = np.zeros(m)
        #     n+= m
            
        self.b = self.__check_shape(b, dim=(m,), varname="b")
        self.c = self.__check_shape(c, dim=(n,), varname="c")
        self.H = self.__check_shape(H, dim=(n, n), varname="H")
        self.M = self.__check_shape(M, dim=(m, m), varname="M")

        rank = np.linalg.matrix_rank(np.block([self.A, -self.M]), tol=self.tol)
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
        
        # init deltas
        self.dx = np.zeros(n)
        self.dy = np.zeros(m)

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

    #attualmente utilizzata
    def set_initial_active_set_from_lu (self):
        """

        """
        def crash (A, xl, xu, cl, cu):
            rowState = cl < cu -self.tol #le righe di A (quindi i valori di s) che sono liberi, e quindi possono far parte della base
            m, n = A.shape
            colState = np.zeros(n, dtype=bool)
            pivot = np.zeros(m)
            pivot[rowState] = 2 #righe già pivotate dalla matrice identità che sarà.
            nBase = sum(rowState) #conto il numero di elementi che sono a true qui, questa è la (temporanea) dimensione della Base. Dobbiamo raggiungere m

            for j in range(n):
                if ~colState[j] or np.allclose(xl[j], xu[j], atol=self.tol): #se abbiamo già trovato la colonna, oppure i valori di x sono fissi, skippiamo
                    continue

                Acol = np.argwhere(np.allclose(A[:, j], 0, atol=self.tol))  #prendiamo i valori diversi da 0 nella matrice.
                if Acol.size == 0: #colonna piena di 0 o si avvicina.
                    continue

                Amax = max (abs (A[Acol, j]))
                Atol = Amax * 0.1
                Atest = abs(A[Acol, j]) > Atol#recupero solo gli elementi che sono significativi
                nz = Atest.size
                npiv = 0 #il numero di pivot che sono stati trovati a questo giro
                ipiv = -np.ones(2) #l'indice degli ultimi pivot presi, default è -1 perché gli array iniziano da 0
                Apiv = np.zeros(2) #il valore dell'ultimo pivot della colonna trovato
                
                for i in range(nz):
                    if (Atest[i]):
                        Ai = abs(A[i,j])
                        ip = pivot[i] #controllo se su quella riga c'è già un pivot o se è possibile prenderne uno.
                        if (ip < 2): #2 indica un pivot perfetto, mentre 0 è non ancora trovato, mentre 1 è trovato ma con valori sotto.
                            if Apiv(ip) < Ai: #controllo se il valore che abbiamo salvato del pivot a quella riga è minore del valore che abbiamo appena trovato
                                Apiv[ip] = Ai #se si me lo salvo e diventa il prossimo concorrente come pivot per quella riga
                                ipiv[ip] = i
                        else: #ho trovato valore su una riga che ha già un pivot
                            npiv += 1 #mi segno che abbiamo un pivot fatto bene

                if (ipiv[0] == -1 and npiv == 0): #Se non abbiamo trovato pivot unmarked e il numero di pivot preesistenti è 0,
                    i = ipiv[1] #allora usiamo le righe che hanno già un pivot ma di cui ne abbiamo trovato di migliore
                else:
                    i = ipiv[0] #se no andiamo di quella unmarked. Ricordiamo che npiv è 0 solo se non sono stati trovati pivot da 3, quindi fissi.
                    #vogliamo quindi priotizzare di prendere pivot che non sono stati già presi invece di aggiornare vecchi.

                if i >= 0: #abbiamo un pivot figo da usare.
                    pivot[i] = 2 #lo mettiamocome usato
                    colState[j] = True
                    nBase += 1 #abbiamo trovato un altro elemento e quindi aggiorniamo la dimensione.

                    if (nBase >= m):
                        break

                    for i in range(nz):
                        if (Atest[i]):
                            Ai = abs(A[i,j])
                            if (Ai > Atol):
                                if (pivot[i] == 0):
                                    pivot[i] = 1 #aggiorno il pivot perché so che ci saranno valori che potrebbero disturbare sotto di lui. (o sopra)
            #padding

            if (rowState != None):
                for i in range(m-nBase): #se ci sono ancora elementi che possiamo aggiungere alla base, allora li aggiungiamo dalla matrice identità.
                    if (pivot[i] < 2): #non è stata presa come colonna di A, allora la prendiamo come riga, quindi come colonna della matrice identità.
                        nBase += 1
                        rowState[i] = True
                        if (nBase >= m):
                            break

            return np.concatenate((colState, rowState))

        #self.B = crash(self.A[:, :self.a_size], self.l[:self.a_size], self.u[:self.a_size], self.l[self.a_size:], self.u[self.a_size:])
        m, n = self.A.shape
        # self.B.fill(False)
        # print(self.A)
        # for i in range(m):
        #     pivots = np.argwhere(self.A[i] != 0)
        #     for pivot in pivots:
        #         if self.B[pivot] == False:
        #             print(pivot)
        #             self.B[pivot] = True
        #             break
            
        # print(self.A[:, self.B])
        # print(sum(self.B) == m)

        self.B[:m] = True
            
        self.N = ~self.B
        self.logger.info(f"B:\n{self.B}\nN:\n{self.N}")

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
        self.dz_l.fill(0)
        self.dz_u.fill(0)

    def get_solution(self):
        """! Function that return the solution of the Quadratic Problem

        @return The result of the problem [ cx + 0.5x.THx + 0.5y.TMy ] with the current solution that satisfy the constrains [Ax + My = b ] and [x >= 0]
        """
        constraint_AMb = self.A @ self.x + self.M @ self.y - self.b
        assert np.allclose(norm_2(constraint_AMb), 0, atol=self.tol), f"AMb: {constraint_AMb}"
        constraint_x = (self.x >= self.l-self.tol) & (self.x <= self.u+self.tol)
        assert np.allclose(constraint_x, True, atol=self.tol), "x: {constraint_x}"
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
                       self.A[:, self.B].T @ self.y + self.z_l[self.B] - self.z_u[self.B])
        self.logger.info(f"H[bb]x[b]: + H[bn]x[n]:\n + c[b] - A[b].Ty + z_l[b] - z_u[b]: {condition_2}") 
        assert np.allclose(norm_2(condition_2), 0, atol=self.tol), condition_2
        condition_3 = (self.H[self.B, :][:, self.N].T @ self.x[self.B] + 
                       self.H[self.N, :][:, self.N]   @ self.x[self.N] + self.c[self.N] - 
                       self.A[:, self.N].T            @ self.y         + self.z_l[self.N] - self.z_u[self.N])
        self.logger.info(f"H[bn].Tx + H[nn]x + c[n] - A[n].Ty + z_l[n] - z_u[n]: {condition_3}") 
        assert np.allclose(norm_2(condition_3), 0, atol=self.tol), condition_3

        #lower bound conditions, 
        condition_4_l = self.z_l[self.B] + self.r_l[self.B]
        self.logger.info(f"z_l[b] + r_l[b]: {condition_4_l}")
        if (not relaxed):
            assert np.allclose(condition_4_l, 0, atol=self.tol), condition_4_l #normal condition
        else:
            assert np.all(condition_4_l <= 0.+self.tol), condition_4_l #relaxed condition
        condition_5_l = np.allclose(self.x[self.N] + self.q_l[self.N], self.l[self.N], atol=self.tol)
        self.logger.info(f"x[n] + q_l[n]: {condition_5_l}")
        condition_6_l = (self.x[self.B] + self.q_l[self.B] >= self.l[self.B]-self.tol)

        #upper bound conditions
        condition_4_u = self.z_u[self.B] + self.r_u[self.B]
        self.logger.info(f"z_u[b] + r_u[b]: {condition_4_u}")
        if (not relaxed):
            assert np.allclose(condition_4_u, 0., atol=self.tol), condition_4_u #normal condition
        else:
            assert np.all(condition_4_u <= 0.+self.tol), condition_4_u #relaxed condition
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
        condition_1 = self.H @ self.x + self.c - self.A.T @ self.y + self.z_l - self.z_u
        self.logger.info(f"Hx + c - A.Ty + z_l - z_u: {condition_1}") 
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
                self.logger.info(f"z_l:\n{self.z_l}")
                self.logger.info(f"z_u:\n{self.z_u}")
                return (
                    self.x,
                    self.y,
                    self.z_l,
                    self.z_u
                )  # non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            #l = l_list[0]
            print(f"{l_list[np.argmin(self.z_l[l_list] + self.r_l[l_list])]}")
            print(f"{l_list[np.argmin((self.z_u[l_list] + self.r_u[l_list]))]}")
            l1 = l_list[np.argmin(self.z_l[l_list] + self.r_l[l_list])]
            l2 = l_list[np.argmin((self.z_u[l_list] + self.r_u[l_list]))]
            if self.z_l[l1] + self.r_l[l1] < (self.z_u[l2] + self.r_u[l2]):
                l = l1
            else:
                l = l2
            
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
        
        self.dx[l] = 1. #if (self.x[l] - self.u[l] - self.q_u[l]) < 0-self.tol else -1. #se siamo già al bound con l non proviamo ad avanzare
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
        if (self.x[l] - self.u[l] - self.q_u[l] > 0.-self.tol):
            self.dx = -self.dx
            self.dy = -self.dy
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
            #self.dz_l[self.N] = tmp_dz
            self.dz_l[l] = -tmp_dz_l
            #self.dx[l] = 1.

            alpha_opt = math.inf if np.allclose(self.dz_l[l], 0, atol=self.tol) else -(self.z_l[l] + self.r_l[l]) / self.dz_l[l]
    
        if (self.z_u[l] + self.r_u[l] < 0.-self.tol): #questi due if, in teoria, sono uno l'opposto dell'altro. o vale uno o vale l'altro.
            #self.dz_u[self.N] = tmp_dz
            self.dz_u[l] = tmp_dz_l
            #tmp_dz = -tmp_dz
            #self.dx[l] = -1.

            alpha_opt = math.inf if np.allclose(self.dz_u[l], 0, atol=self.tol) else (self.z_u[l] + self.r_u[l]) / self.dz_u[l]

        self.dz_l[self.N] = np.where(self.z_l[self.N] + self.r_l[self.N] < 0-self.tol, -tmp_dz, 0)
        self.dz_u[self.N] = np.where(self.z_u[self.N] + self.r_u[self.N] < 0-self.tol, tmp_dz, 0)
        self.logger.info(f"delta z_l\n{self.dz_l}")
        self.logger.info(f"delta z_u\n{self.dz_u}")

        min_mask = ~np.isclose(self.dx, 0, atol=self.tol)
        to_min = np.where(self.dx < 0, (self.x - self.l + self.q_l), (self.x - self.u - self.q_u))
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
        self.logger.info(f"to_min:\n{to_min}\n")
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
            
        self.x        += alpha * self.dx
        self.y        += alpha * self.dy
        self.z_l      += alpha * self.dz_l
        self.z_u      += alpha * self.dz_u
        
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
        print(self.dx[l])
        self.logger.info(f"delta y:\n{self.dy}")

        tmp_dz = (
              self.H[self.N, l]           * self.dx[l] #scalare 
            + self.H[self.B][:, self.N].T @ self.dx[self.B] #moltiplicazione #Nx#B per #Bx1
            - self.A[:, self.N].T         @ self.dy[:] # moltiplicazione #Nxm per #mx1
        )

        if (self.z_l[l] + self.r_l[l] < 0.-self.tol):
            #self.dz_l[self.N] = tmp_dz
            self.dz_l[l] = -1.
            
            alpha_opt = -(self.z_l[l] + self.r_l[l])
            
        if (self.z_u[l] + self.r_u[l] < 0.-self.tol):
            #self.dz_u[self.N] = tmp_dz
            self.dz_u[l] = 1.
            print(self.z_u[l])
            
            alpha_opt = -(self.z_u[l] + self.r_u[l])

        self.dz_l[self.N] = np.where(self.z_l[self.N] + self.r_l[self.N] < 0-self.tol, -tmp_dz, 0)
        self.dz_u[self.N] = np.where(self.z_u[self.N] + self.r_u[self.N] < 0-self.tol, tmp_dz, 0)
        self.logger.info(f"delta z_l\n{self.dz_l}")
        self.logger.info(f"delta z_u\n{self.dz_u}")
        
        min_mask = ~np.isclose(self.dx, 0, atol=self.tol)
        to_min = np.where(self.dx < 0, (self.x - self.l + self.q_l), -(self.x - self.u - self.q_u))
        print(to_min)
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-self.dx[min_mask]
        print(f"z_u: {self.z_u[min_mask]} r_u: {self.r_u[min_mask]}")
        print(f"z_l: {self.z_l[min_mask]} r_l: {self.r_l[min_mask]}")
        self.logger.info(f"to_min:\n{to_min[self.B]}")
        k = np.argmin(to_min)
        print(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( opt = {alpha_opt}; max = {alpha_max});")
        
        if np.isclose(alpha, 0, atol=self.tol):
            self.logger.exception(f"Step size is zero")
            raise Exception("Step size is zero")
        
        self.x        += alpha * self.dx
        self.y        += alpha * self.dy
        self.z_l      += alpha * self.dz_l
        self.z_u      += alpha * self.dz_u
        
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
            l_list = np.argwhere(((self.x - self.l + self.q_l) < 0.-self.tol) | ((self.x - self.u - self.q_u) > 0.+self.tol))
            self.logger.info(f"The indexes that violate the constrains are: {l_list.flatten()}")
            if l_list.size == 0:
                self.logger.info(f"The dual algorith just terminated its course. The solutions are as follows:")
                self.logger.info(f"x:\n{self.x}")
                self.logger.info(f"y:\n{self.y}")
                self.logger.info(f"z_l:\n{self.z_l}")
                self.logger.info(f"z_u:\n{self.z_u}")
                return (
                    self.x,
                    self.y,
                    self.z_l,
                    self.z_u
                )  # non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            #l = l_list[0]
            #sarebbe da fare un secondo argmin con l_list[argmin] dei due argmin interni ma tbh fino a che va bene, va bene così
            print(f"{l_list[np.argmin(self.x[l_list] - self.l[l_list] + self.q_l[l_list])]}")
            print(f"{l_list[np.argmax(self.x[l_list] - self.u[l_list] - self.q_u[l_list])]}")
            l1 = l_list[np.argmin(self.x[l_list] - self.l[l_list] + self.q_l[l_list])]
            l2 = l_list[np.argmax(self.x[l_list] - self.u[l_list] - self.q_u[l_list])]
            if (self.x[l1] - self.l[l1] + self.q_l[l1] < -(self.x[l2] - self.u[l2] - self.q_u[l2])):
                l = l1
            else:
                l = l2

            if (self.B[l] == True):
                self.B[l] = False  # prendo il primo elemento di l e lo levo da N.
                self.dual_base(l)
                self.reset_deltas()
            else:
                self.N[l] = False
            
            while (self.x[l] - self.l[l] + self.q_l[l]) < 0.-self.tol or (self.x[l] - self.u[l] - self.q_u[l]) > 0.+self.tol :
                #input("premi per continuare...")
                self.dual_intermediate(l)
                self.reset_deltas()
            self.N[l] = True

    def dual_base(self, l):
        """! Function that do the base step of the Dual problem

        @param l the index that violate the constrain
        """
        self.logger.info(f"-"*20)
        self.logger.info(f"Base step of the Dual problem, done with the index: {l}")

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
        
        if (self.x[l] - self.l[l] + self.q_l[l] < 0.-self.tol):
            self.dz_l[l] = 1. #in teria dovrebbe essere -1 ma non torna :/
            #self.dz_l[self.N] = tmp_dz

            alpha_opt = np.inf if np.allclose(self.dx[l], 0, atol=self.tol) else -(self.x[l] - self.l[l] + self.q_l[l])/self.dx[l] #dx > 0
            
        elif (-self.x[l] + self.u[l] + self.q_u[l] < 0.-self.tol):
            self.dz_u[l] = 1.
            #self.dz_u[self.N] = tmp_dz
            
            alpha_opt = np.inf if np.allclose(self.dx[l], 0, atol=self.tol) else -(-self.x[l] + self.u[l] + self.q_u[l])/self.dx[l] #dx > 0

        self.dz_l[self.N] = np.where(self.z_l[self.N] + self.r_l[self.N] < 0-self.tol, -tmp_dz, 0)
        self.dz_u[self.N] = np.where(self.z_u[self.N] + self.r_u[self.N] < 0-self.tol, tmp_dz, 0)
        self.logger.info(f"delta z_l\n{self.dz_l}")
        self.logger.info(f"delta z_u\n{self.dz_u}")

        min_mask = ( self.dz_u < 0 ) | ( self.dz_l < 0)
        to_min = (self.r_u + self.z_u) - (self.r_l + self.z_l)
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-(self.dz_u[min_mask] - self.dz_l[min_mask])

        print(f"min_mask: {min_mask}")
        print(f"z_u: {self.z_u[min_mask]} r_u: {self.r_u[min_mask]}")
        print(f"z_l: {self.z_l[min_mask]} r_l: {self.r_l[min_mask]}")
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
        
        self.x        += alpha * self.dx
        self.y        += alpha * self.dy
        self.z_l      += alpha * self.dz_l
        self.z_u      += alpha * self.dz_u
        
        if (self.x[l] - self.l[l] + self.q_l[l]) < 0.-self.tol or (self.x[l] - self.u[l] - self.q_u[l]) > 0.+self.tol:
            self.logger.info(f"k: {k}")
            self.B[k] = True
            self.N[k] = False
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z_l:\n{self.z_l}")
        self.logger.info(f"z_u:\n{self.z_u}")
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

        if (self.x[l] - self.l[l] + self.q_l[l] < 0.-self.tol):
            #self.dz_l[self.N] = tmp_dz
            self.dz_l[l] = -tmp_dz_l
            #self.dx[l] = 1.
            
            alpha_opt = (self.x[l] - self.l[l] + self.q_l[l])

        if (self.x[l] - self.u[l] + self.q_u[l] > 0.+self.tol):
            #self.dz_u[self.N] = tmp_dz
            self.dz_u[l] = tmp_dz_l
            #self.dx[l] = -1.

            alpha_opt = (-self.x[l] + self.u[l] + self.q_u[l])
        
        self.logger.info(f"delta x:\n{self.dx}")
        self.logger.info(f"delta y:\n{self.dy}")
        self.dz_l[self.N] = np.where(self.z_l[self.N] + self.r_l[self.N] < 0-self.tol, -tmp_dz, 0)
        self.dz_u[self.N] = np.where(self.z_u[self.N] + self.r_u[self.N] < 0-self.tol, tmp_dz, 0)
        self.logger.info(f"delta z_l\n{self.dz_l}")
        self.logger.info(f"delta z_u\n{self.dz_u}")

        min_mask = ( self.dz_u < 0 ) | ( self.dz_l < 0)
        to_min = (self.r_u + self.z_u) - (self.r_l + self.z_l)
        to_min[~min_mask] = np.inf
        to_min[min_mask] = to_min[min_mask]/-(self.dz_u[min_mask] - self.dz_l[min_mask])

        print(f"min_mask: {min_mask}")
        print(f"z_u: {self.z_u[min_mask]} r_u: {self.r_u[min_mask]}")
        print(f"z_l: {self.z_l[min_mask]} r_l: {self.r_l[min_mask]}")
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

        self.x        += alpha * self.dx
        self.y        += alpha * self.dy
        self.z_l      += alpha * self.dz_l
        self.z_u      += alpha * self.dz_u
        
        if (self.x[l] - self.l[l] + self.q_l[l]) < 0.-self.tol or (self.x[l] - self.u[l] - self.q_u[l]) > 0.+self.tol: #effettivamente anche alpha == alpha_max ha senso
            self.logger.info(f"k: {k}")
            self.B[k] = True
            self.N[k] = False
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z_l:\n{self.z_l}")
        self.logger.info(f"z_u:\n{self.z_u}")
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
    u = 7 * np.ones(n)
    l = np.zeros(n)
    qp = quadratic_problem (A, b, c, H, M, l=l, u=u, verbose = True)

    B = (np.random.rand(n) - np.random.rand(n)) > 0
    N = ~B
#    qp.set_initial_active_set_from_lu()
    print(qp.solve())
    pass
