#!bin/usr/python
# coding=utf-8

import numpy as np
import math


class quadratic_problem:
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

    def __init__(self, A, b, c, H, M, q=None, r=None, tol=1e-8):
        def check_shape(x, dim=None, varname=""):
            if isinstance(x, np.ndarray):
                if dim == None or x.shape == dim:
                    return x
                else:
                    raise TypeError(f'<{varname}> has shape <{x.shape}> expected <{dim}>')
            else:
                raise TypeError(f'<{varname}> is not {type({np.ndarray})}')

        self.tol = tol
        
        # shape of A is assumed to be correct
        self.A = check_shape(A, varname="A")
        m, n = A.shape

        self.b = check_shape(b, dim=(m,), varname="b")
        self.c = check_shape(c, dim=(n,), varname="c")
        self.H = check_shape(H, dim=(n, n), varname="H")
        self.M = check_shape(M, dim=(m, m), varname="M")

        self.q = check_shape(q, dim=(n,), varname="q") if q else np.zeros(n)
        self.r = check_shape(r, dim=(n,), varname="r") if r else np.zeros(n)

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
        print("-"*20)
        print("Starting the resolution of the Primal Problem of the Active Sets")
        print()

        # --------------- parte in cui si decidono gli x, y, z e B e N
        self.x = x
        self.y = y
        self.z = z
        sum_xq = self.x + self.q
        sum_zr = self.z + self.r
        print ("sum_xq: ", sum_xq)
        print ("sum_zr: ", sum_zr)
        self.N = (
            sum_xq
        ) == 0  # dovrebbe restituire un vettore di veri e falsi. Also da fare che non sia == 0 ma in una certa epsilon.
        self.B = (
            sum_zr
        ) == 0  # come sopra
        print("N: ", self.N)
        print("B, ", self.B)
        # --------------- controllo che abbiamo preso cose sensate
        if all(np.logical_xor(self.B, self.N)) == False:
            print ("Errore nella scelta di x, y, z")
            return None  # ci sono degli elementi che sono sia in N che in B.
        if any((sum_xq) < 0):
            print ("Errore nella scelta di x, y, z, ma con la somma")
            return None  # non soddisfa una delle condizioni.
        # --------------- Inizio loop principale
        while True:
            l_list = np.argwhere(sum_zr < 0)[0]
            print ("l: ", l_list)
            if l_list == []:
                print ("Processo terminato")
                print ("x: ", self.x)
                print ("y: ", self.y)
                print ("z: ", self.z)
                return (
                    self.x,
                    self.y,
                    self.z,
                )  # non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            l = l_list[0]
            self.N[l] = False  # prendo il primo elemento di l e lo levo da N.
            self.primal_base(l)
            while (z[l] + r[l]) < 0:
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
        print ("-"*20)
        print ("Iterazione base del problema usando l'indice: ", l)
        print ()
        
        self.dx[l] = 1
        B_size = np.sum((self.B))
        
        print ("Hbb:\n", self.H[self.B, :][:, self.B])
        print ("Ab:\n", self.A[:, self.B])
        print ("M:\n", self.M)
        K_I = np.block ([
            [self.H[self.B, :][:, self.B], self.A[:, self.B].T],
            [self.A[:, self.B],           -self.M             ],
        ])
        print ("K_I:\n", K_I)
        print ()

        print ("Hb:\n", self.H[self.B][:, l])
        print ("Al:\n", self.A[l])
        tmp_b = -np.concatenate(
            (self.H[self.B][:, l], self.A[l]),
            axis=0
        )
        print ("b:\n", tmp_b)
        print ()
        
        tmp_sol = np.linalg.solve(K_I, tmp_b) #da cambiare
        print ("sol:\n", tmp_sol)
        print ()
        
        self.dx[self.B], self.dy[:] = tmp_sol[:B_size], -tmp_sol[B_size:]

        self.dz[self.N] = (
            self.H[self.N, l] * self.dx[l] #qui * va bene perché delta_x_l è uno scalare
            + np.matmul(self.H[self.B][:, self.N].T, self.dx[B]) #qui usiamo matmul perché è la moltiplicazione di una matrice #Nx#B per un vettore #Bx1
            - np.matmul(self.A[:, self.N].T, self.dy) #come sopra, #Nxm per mx1
        )
        self.dz[l] = (
            self.H[l, l] * self.dx[l] #scalare
            + np.matmul(self.H[self.B][:, l].T, self.dx[self.B]) # qui è un vettore 1x#B per #B
            - np.matmul(self.A[l].T, self.dy) # qui è 1xm per mx1
        )
        print("delta z\tN")
        print(self.dz, "\t", self.N)

        alpha_opt = math.inf if np.allclose(dx[l], 0, rtol=self.tol) else -(self.z[l] + self.r[l]) / self.dz[l]

        min_mask = ( self.dx < 0 )
        to_min = (self.x[min_mask] + self.q[min_mask]) / -self.dx[min_mask]
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        print ("alpha = min (", alpha_opt, "; ", alpha_max, "); k = ", k)
        print ()
        
        if np.isinf(alpha):
            raise Exception("Primal is Unboundend (Dual is unfeasible)")  # il problema è impraticabile

        self.x[l] += alpha * self.dx[l]
        self.x[B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[N] += alpha * self.dz[self.N]
        if z[l] + r[l] < 0:
            self.B[k] = False
            self.N[k] = True
        
        print ("x:\n", x)
        print ("y:\n", y)
        print ("z:\n", z)
        print ("B:\n", B)
        print ("N:\n", N)
        print ()
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
        print("-"*20)
        print("Passo intermendio del problema usando l'indice: ", l)
        print()
        
        self.dz[l] = 1
        B_size = np.sum((self.B))

        print("Hll:\n", self.H[l, l])
        print("HBl:\n", self.H[self.B][:, l])
        print("Hbb:\n", self.H[self.B][:, self.B])
        print("Al:\n", self.A[l, :])
        print("Ab:\n", self.A[self.B, :])
        print("M:\n", self.M)
        K_I = np.block([
            [self.H[l, l],         self.H[self.B, l].T,       self.A[l, :]    ],
            [self.H[self.B][:, l], self.H[self.B][:, self.B], self.A[self.B].T],
            [self.A[l, :],         self.A[self.B, :],         -self.M]
        ])
        print("K_I:\n", K_I)
        print()
        
        tmp_b = np.concatenate(
            np.ones(1),
            np.zeros(B_size + self.y.size)
        )
        print("b:\n", tmp_b)
        print()
        
        tmp_sol = np.linalg.solve(K_I, tmp_b)
        print ("sol: ", tmp_sol)
        print ()
        # robo da risolvere
        self.dx[l], self.dx[self.B], self.dy[:] = tmp_sol[0], tmp_sol[1: B_size +1], -tmp_sol[B_size + 1:]

        self.dz[N] = (
            self.H[self.N][:, l] * self.dx[l] #scalare 
            + np.matmul(self.H[self.B][:, self.N].T, self.dx[self.B]) #moltiplicazione #Nx#B per #Bx1
            - np.matmul(self.A[:, self.N].T, self.dy) # moltiplicazione #Nxm per #mx1
        )
        print("delta z\tN")
        print(self.dz, "\t", self.N)
        print()
        
        alpha_opt = -(self.z[l] + self.r[l])
        min_mask = ( self.dx < 0 )  # in realtà da rivedere perché questo è di dimensione #B, quindi c'è da aumentare sta maschera o qualcosa del genere
        to_min = (self.x[min_mask] + self.q[min_mask]) / -self.dx[min_mask]
        k = np.argmin(to_min)
        alpha_max = to_min[k]

        alpha = min(alpha_opt, alpha_max)
        print ("alpha = min (", alpha_opt, "; ", alpha_max, "); k = ", k)
        print()
        
        self.x[l] += alpha * self.dx[l]
        self.x[B] += alpha * self.dx[self.B]
        self.y    += alpha * self.dy
        self.z[l] += alpha * self.dz[l]
        self.z[N] += alpha * self.dz[N]
        if z[l] + self.r[l] < 0:
            self.B[k] = False
            self.N[k] = True
        print ("x:\n", x)
        print ("y:\n", y)
        print ("z:\n", z)
        print ("B:\n", B)
        print ("N:\n", N)
        print ()
        return

    # kind of a mess but ok


if __name__ == "__main__":
    A = np.array([ [1, 1], [2, 1] ])
    b = np.array([2, 3])
    c = np.array([2, 3])
    H = np.array([ [-2, 0], [0, -2] ])
    M = np.zeros((2, 2))
    qp = quadratic_problem (A, b, c, H, M)
    x = np.array([1, 0])
    y = np.zeros(2)
    z = np.array([0, -1])
    qp.primal_active_set(x, y, z)
    pass
