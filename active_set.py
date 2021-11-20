#!bin/usr/python
# coding=utf-8

import numpy as np
import math
import logging
import sys


class quadratic_problem:

    @staticmethod
    def __check_shape(x, dim=None, varname=""):
        if isinstance(x, np.ndarray):
            if dim == None or x.shape == dim:
                return x
            else:
                raise TypeError(f'<{varname}> has shape <{x.shape}> expected <{dim}>')
        else:
            raise TypeError(f'<{varname}> is not {type({np.ndarray})}')
    """
    Costruttore del problema quadratico del tipo:
    min(x, y) = c.T * x + 1/2 * x.T * H * x + 1/2 * y.T * M * y 
    dati: A * x + M * y = b, x >= 0

    consideriamo che
    x = R(n)
    y = R(m)
    z = R(n)
    params:
     - A La matrice dei vincoli R(m x n)
     - b il vettore dei vincoli R(m)
     - c il vettore R(n)
     - H la matrice di del problema, di solito una matrice hessiana. è positiva, simmetrica e semidefinita R(n x n)
     - M la matrice relativa alla seconda variabile. Se è 0, il problema è un problema convesso quadratico convenzionale. R(m x m)
     - q il vettore di vincoli per il problema primale R(n). default=0
     - r il vettore di vincoli per il problema duale R(n). default=0
    """

    def __init__(self, A, b, c, H, M, q=None, r=None, tol=1e-8, verbouse=False):
        self.logger = logging.getLogger('execution_logger')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler ("execution.log")
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
#        logging.basicConfig (filename = 'execution.log', encoding='utf-8', level = logging.DEBUG)
        if verbouse:
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
        
    """
    funzione per calcolare il problema primale degli active set

    params:
     - (x, y, z): tripla di punti su cui iniziare l'algoritmo
    returns:
     - (x, y, z): tripla di punti ottimali della soluzione
    """
    def primal_active_set(self, x, y, z):
        self.logger.info("-"*20)
        self.logger.info(f"Starting the resolution of the Primal Problem of the Active Sets")

        # --------------- parte in cui si decidono gli x, y, z e B e N
        self.x = x
        self.y = y
        self.z = z
        sum_xq = self.x + self.q
        sum_zr = self.z + self.r
        
        self.logger.info(f"sum_xq: {sum_xq}")
        self.logger.info(f"sum_zr: {sum_zr}")

        self.N = (
            sum_xq
        ) == 0  # dovrebbe restituire un vettore di veri e falsi. Also da fare che non sia == 0 ma in una certa epsilon.
        self.B = (
            sum_zr
        ) == 0  # come sopra
        
        self.logger.info(f"N:\n{self.N}")
        self.logger.info(f"B:\n{self.B}")
        # --------------- controllo che abbiamo preso cose sensate
        if all(np.logical_xor(self.B, self.N)) == False:
            self.logger.error("Error in the choice of x, y, z: they don't respect the conditions")
            return None  # ci sono degli elementi che sono sia in N che in B.
        if any((sum_xq) < 0):
            self.logger.error("Error in the choice of x, y, z: the sum of x and q doesn't make sense")
            return None  # non soddisfa una delle condizioni.
        # --------------- Inizio loop principale
        while True:
            l_list = np.argwhere((self.z - self.r) < 0)
            print ("l: ", l_list.flatten())
            if l_list.size == 0:
                self.logger.info(f"The primal algorith just terminated his course. The solutions are as follows:")
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
            while (self.z[l] + self.r[l]) < 0:
                #input("premi per continuare...")
                self.primal_intermediate(l)
            self.B[l] = True

    """
    Funzione di inizializzazione del primale degli active set.

    params:
    - B: la maschera degli indici della base nella forma [b1, b2, ..., bn] dove bi = [treu/false]
    - N: la maschera degli indici normali nella forma [n1, n2, ..., nm] dove n1 = [true/false]
    - l: l'indice che è stato scelto, e che attualmente non è ne in B ne in N
    - x, y, z: la tripla di punti di cui dobbiamo trovare la nuova iterazione

    returns: (B, N, x, y, z) i valori aggiornati
    """
    def primal_base(self, l):
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
        self.logger.info(f"detla y:\n{self.dy}")

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
        if np.sum(min_mask) == 0:
            alpha_max = np.inf
        else:
            to_min = (self.x[min_mask] + self.q[min_mask]) / -self.dx[min_mask]
            k = np.argmin(to_min)
            alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ({alpha_opt}; {alpha_max});")
        
        if np.isinf(alpha):
            self.logger.exception(f"Primal is Unbounded (Dual is unfeasible")
            raise Exception("Primal is Unboundend (Dual is unfeasible)")  # il problema è impraticabile

        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[self.N]
        
        if self.z[l] + self.r[l] < 0: #effettivamente anche alpha == alpha_max ha senso
            self.logger.info(f"k:{k}")
            self.B[k] = False
            self.N[k] = True
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z:\n{self.z}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

    """
    Il passo intermedio che viene fatto dal problema primale

    params:
    - B: la maschera degli indici della base nella forma [b1, b2, ..., bn] dove bi = [treu/false]
    - N: la maschera degli indici normali nella forma [n1, n2, ..., nm] dove n1 = [true/false]
    - l: l'indice che è stato scelto, e che attualmente non è ne in B ne in N 
    - x, y, z: la tripla di punti di cui dobbiamo trovare la nuova iterazione                                                    

    returns: (B, N, x, y, z) i valori aggiornati
    """

    def primal_intermediate(self, B, N, l, x, y, z):
        self.logger.info(f"-"*20)
        self.logger.info(f"Passo intermendio del problema usando l'indice: {l}")
        
        self.dz[l] = 1
        B_size = np.sum((self.B))

        self.logger.info(f"Hll:\n{self.H[l, l]}")
        self.logger.info(f"HBl:\n{self.H[self.B][:, l]}")
        self.logger.info(f"Hbb:\n{self.H[self.B][:, self.B]}")
        self.logger.info(f"Al:\n{self.A[l, :]}")
        self.logger.info(f"Ab:\n{self.A[self.B, :]}")
        self.logger.info(f"M:\n{self.M}")
        K_I = np.block([
            [self.H[l, l],         self.H[self.B, l].T,       self.A[l, :]    ],
            [self.H[self.B][:, l], self.H[self.B][:, self.B], self.A[self.B].T],
            [self.A[l, :],         self.A[self.B, :],         -self.M]
        ])
        self.logger.info(f"K_I:\n{K_I}")
        
        tmp_b = np.concatenate(
            np.ones(1),
            np.zeros(B_size + self.y.size)
        )
        self.logger.info(f"b:\n{tmp_b}")
        
        tmp_sol = np.linalg.solve(K_I, tmp_b)
        self.logger.info(f"sol:\n{tmp_sol}")
        # robo da risolvere
        self.dx[l], self.dx[self.B], self.dy[:] = tmp_sol[0], tmp_sol[1: B_size +1], -tmp_sol[B_size + 1:]
        self.logger.info(f"delta x:\n{self.dx}")
        self.logger.info(f"delta y:\n{self.dy}")

        self.dz[N] = (
              self.H[self.N][:, l]        * self.dx[l] #scalare 
            + self.H[self.B][:, self.N].T @ self.dx[self.B] #moltiplicazione #Nx#B per #Bx1
            - self.A[:, self.N].T         @ self.dy # moltiplicazione #Nxm per #mx1
        )
        self.logger.info(f"delta z\n{self.dz}")
        
        alpha_opt = -(self.z[l] + self.r[l])
        min_mask = ( self.dx < 0 )  # in realtà da rivedere perché questo è di dimensione #B, quindi c'è da aumentare sta maschera o qualcosa del genere
        if np.sum(min_mask) == 0:
            alpha_max = np.inf
        else:
            to_min = (self.x[min_mask] + self.q[min_mask]) / -self.dx[min_mask]
            k = np.argmin(to_min)
            alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        self.logger.info(f"alpha = min ( {alpha_opt}; {alpha_max});")
        
        self.x[l] += alpha * self.dx[l]
        self.x[self.B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[self.N] += alpha * self.dz[N]
        
        if self.z[l] + self.r[l] < 0:
            self.logger.info(f"k:\n{k}")
            self.B[k] = False
            self.N[k] = True
        
        self.logger.info(f"x:\n{self.x}")
        self.logger.info(f"y:\n{self.y}")
        self.logger.info(f"z:\n{self.z}")
        self.logger.info(f"B:\n{self.B}")
        self.logger.info(f"N:\n{self.N}")
        return

    # kind of a mess but ok


if __name__ == "__main__":
    n, m = 4, 6
    A = np.eye(m, n)
    M = np.eye(m, m)
    H = np.eye(n, n)
    b = np.array([ 1., 1., 2., 2., 0., 0.])
    c = np.array([-1.,-1., 1., 1.])
    qp = quadratic_problem (A, b, c, H, M)
    x = np.array([1., 1., 0., 0.])
    y = np.array([0., 0., 2., 2., 0., 0.])
    z = np.array([0., 0.,-1.,-1.])
    qp.primal_active_set(x, y, z)
    pass
