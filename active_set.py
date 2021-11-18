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

    def __init__(self, A, b, c, H, M, q=None, r=None):
        def check_shape(x, dim=None, varname=""):
            if isinstance(x, np.ndarray):
                if dim == None or x.shape == dim:
                    return x
                else:
                    raise TypeError(f"{varname} has shape {x.shape} expected {dim}")
            else:
                raise TypeError(f"{varname} is not {type({np.ndarray})}")

        # shape of A is assumed to be correct
        self.A = check_shape(A, varname="A")
        m, n = A.shape

        self.b = check_shape(b, dim=(m,), varname="b")
        self.c = check_shape(c, dim=(n,), varname="c")
        self.H = check_shape(H, dim=(n, n), varname="H")
        self.M = check_shape(M, dim=(m, m), varname="M")

        self.q = check_shape(q, dim=(n,), varname="q") if q else np.zeros(n)
        self.r = check_shape(r, dim=(n,), varname="r") if q else np.zeros(n)

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
        
        sum_xq = x + self.q
        sum_zr = z + self.r
        N = (
            sum_xq
        ) == 0  # dovrebbe restituire un vettore di veri e falsi. Also da fare che non sia == 0 ma in una certa epsilon.
        B = (
            sum_zr
        ) == 0  # come sopra
        print("N: ", N)
        print("B, ", B)
        if np.logical_xor(B, N) == False:
            return None  # ci sono degli elementi che sono sia in N che in B.
        if (sum_xq) < 0:
            return None  # non soddisfa una delle condizioni.
        while 1:
            l_array = range(sum_zr.len)[sum_zr < 0]
            if l_array == []:
                print ("Processo terminato")
                print ("x: ", x)
                print ("y: ", y)
                print ("z: ", z)
                return (
                    x,
                    y,
                    z,
                )  # non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            l = l_array[0]
            N[l] = False  # prendo il primo elemento di l e lo levo da N.
            (B, N, x, y, z) = self.primal_base(B, N, l, x, y, z)
            while z[l] + r[l]:
                (N, N, x, y, z) = self.primal_intermediate(B, N, l, x, y, z)
            B[l] = True

    """
    Funzione di inizializzazione del primale degli active set.

    params:
    - B: la maschera degli indici della base nella forma [b1, b2, ..., bn] dove bi = [treu/false]
    - N: la maschera degli indici normali nella forma [n1, n2, ..., nm] dove n1 = [true/false]
    - l: l'indice che è stato scelto, e che attualmente non è ne in B ne in N
    - x, y, z: la tripla di punti di cui dobbiamo trovare la nuova iterazione

    returns: (B, N, x, y, z) i valori aggiornati
    """

    def primal_base(self, B, N, l, x, y, z):
        print("-"*20)
        print("Iterazione base del problema usando l'indice: ", l)
        print()
        
        delta_x_l = 1

        tmp_A = np.concatenate(
            (
                np.concatenate((self.H[B, :][:, B], self.A[B].T), axis=1),
                np.concatenate((self.A[B], -self.M), axis=1),
            ),
            axis=0,
        )  # not sure about this whole thing
        print("K_I: ", tmp_A)
        tmp_b = -np.concatenate(
            self.H[B, l], self.A[:, l]
        )  # wait no sta cosa non mi torna perché sono scalari aaaaaaaaa
        print("b: ", tmp_b)
        tmp_sol = np.linalg.solve(tmp_A, tmp_b)
        print("sol: ", tmp_sol)
        print()
        # tutta sta parte del sistema è precaria e sadda provà
        delta_y = tmp_sol[B.len:]  # i'm not sure, dovrebbe essere di dimensione m
        delta_x_B = tmp_sol[:B.len]

        delta_z_N = (
            self.H[N, l] * delta_x_l #qui * va bene perché delta_x_l è uno scalare
            + np.matmul(self.H[B][:, N].T, delta_x_B) #qui usiamo matmul perché è la moltiplicazione di una matrice #Nx#B per un vettore #Bx1
            - np.matmul(self.A[:, N].T, delta_y) #come sopra, #Nxm per mx1
        )
        print("delta z N: ", delta_z_N)
        delta_z_l = (
            self.H[l, l] * delta_x_l #scalare
            + np.matmul(self.H[B, l].T, delta_x_B) # qui è un vettore 1x#B per #B
            - np.matmul(self.A[l].T, delta_y) # qui è 1xm per mx1
        )
        print("delta z l", delta_z_l)

        alpha_opt = math.inf if delta_z_l == 0 else -(z[l] + r[l]) / delta_z_l
        min_mask = (
            delta_x_B < 0
        )
        # da fare che qui calcoli solo se delta_x è negativo. Però noi abbiamo che non sappiamo delta_x, sooooo
        # nell'intermiedate ho proposto una soluzione ma sono poco sicuro
        alpha_max = np.min((x[B] + self.q[B]) / -delta_x_B)
        k = np.argmin((x[B] + self.q[B]) / -delta_x_B)

        alpha = min(alpha_opt, alpha_max)
        print ("alpha = min (", alpha_opt, "; ", alpha_max, "); k = ", k)
        print ()
        if alpha == math.inf:
            return  # il problema è impraticabile

        x[l] = x[l] + alpha * delta_x_l
        x[B] = x[B] + alpha * delta_x_b
        y = y + alpha * delta_y
        z[l] = z[l] + alpha * delta_z_l
        z[N] = z[N] + alpha * delta_z_N
        if z[l] + r[l] < 0:
            B[k] = False
            N[k] = True
        print ("x: ", x)
        print ("y: ", y)
        print ("z: ", z)
        print ("B: ", B)
        print ("N: ", N)
        print ()
        return B, N, x, y, z

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
        delta_z_l = 1

        tmp_A = np.concatenate(
            (
                np.concatenate((self.H[l, l], self.H[B, l].T, self.A[l, :].T), axis=1),
                np.concatenate((self.H[B][l], self.H[B][:, B], self.A[B].T), axis=1),
                np.concatenate((self.A[l, :], self.A[B], -self.M), axis=1),
            ),
            axis=0,
        )
        tmp_b = np.array([1, 0, 0])
        tmp_sol = np.linalg.solve(tmp_A, tmp_b)
        print ("K_I: ", tmp_A)
        print ("b: ", tmp_b)
        print ("sol: ", tmp_sol)
        print ()
        # robo da risolvere
        delta_x_l = tmp_sol[0] #non sono pienamente sicuro
        delta_x_B = tmp_sol[1 : B.len + 1] #dimensione #B
        delta_y = tmp_sol[B.len + 1 :] #dimensione m

        delta_z_N = (
            self.H[N, l] * delta_x_l #scalare 
            + np.matmul(self.H[B][:, N].T, delta_x_B) #moltiplicazione #Nx#B per #Bx1
            - np.matmul(self.A[:, N].T, delta_y) # moltiplicazione #Nxm per #mx1
        )
        print ("delta z N:", delta_z_N)
        alpha_opt = -(z[l] + self.r[l])
        min_mask = (
            delta_x_B < 0
        )  # in realtà da rivedere perché questo è di dimensione #B, quindi c'è da aumentare sta maschera o qualcosa del genere

        alpha_max = np.min((x[min_mask] + self.q[min_mask]) / -delta_x_B[min_mask])
        k = np.argmin((x[min_mask] + self.q[min_mask]) / -delt_x_B[min_mask])

        alpha = min(alpha_opt, alpha_max)
        print ("alpha = min (", alpha_opt, "; ", alpha_max, "); k = ", k)
        print()
        
        x[l] = x[l] + alpha * delta_l
        x[B] = x[B] + alpha * delta_x_B
        y = y + alpha * delta_y
        z[l] = z[l] + alpha * delta_z_l
        z[N] = z[N] + alpha * delta_z_N
        if z[l] + self.r[l] < 0:
            B[k] = False
            N[k] = True
        print ("x: ", x)
        print ("y: ", y)
        print ("z: ", z)
        print ("B: ", B)
        print ("N: ", N)
        print ()
        return (B, N, x, y, z)

    # kind of a mess but ok


if __name__ == "__main__":
    pass
