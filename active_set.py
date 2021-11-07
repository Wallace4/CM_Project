import numpy as np
import math

class quadratic_problem ():
    """
    Costruttore del problema quadratico del tipo:
    min(x, y) = c.T * x + 1/2 * x.T * H * x + 1/2 * y.T * M * y 
    dati: A * x + M * y = b, x >= 0

    consideriamo che
    x = R(n)
    y = R(m)
    z = R(n)
    params:
     - A La matrice dei vincoli R(n x m)
     - b il vettore dei vincoli R(m)
     - c il vettore R(n)
     - H la matrice di del problema, di solito una matrice hessiana. è positiva, simmetrica e semidefinita R(n x n)
     - M la matrice relativa alla seconda variabile. Se è 0, il problema è un problema convesso quadratico convenzionale. R(m x m)
     - q il vettore di vincoli per il problema primale R(n)
     - r il vettore di vincoli per il problema duale R(n)
    """
    def __init__(self, A, b, c, H, M, q, r):
        self.A = A if isinstance(type(A), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
        self.b = b if isinstance(type(b), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
        self.c = c if isinstance(type(c), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>') 
        self.H = H if isinstance(type(H), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>') 
        self.M = M if isinstance(type(M), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>') 
        self.q = q if isinstance(type(q), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>') 
        self.r = r if isinstance(type(r), np.ndarray()) else TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>') 

    """
    funzione per calcolare il problema primale degli active set

    params:
     - (x, y, z): tripla di punti su cui iniziare l'algoritmo
    returns:
     - (x, y, z): tripla di punti ottimali della soluzione
    """
    def primal_active_set (self, x, y, z):
        sum_xq = x + self.q
        sum_zr = z + self.r
        N = (sum_xq) == 0 #dovrebbe restituire un vettore di veri e falsi. Also da fare che non sia == 0 ma in una certa epsilon.
        B = (sum_zr) == 0 #come sopra
        if (np.logical_xor(B, N) == False): return None #ci sono degli elementi che sono sia in N che in B.
        if (sum_xq) < 0: return None #non soddisfa una delle condizioni.
        while (1):
            l_array = range(sum_zr.len)[sum_zr < 0]
            if l_array == []:
                return x, y, z #non si può fare un passo, aka siamo arrivati alla nostra soluzione ottima
            l = l_array[0]
            N[l] = False #prendo il primo elemento di l e lo levo da N.
            (B, N, x, y, z) = self.primal_base (B, N, l, x, y, z)
            while (z[l] + r[l]):
                (N, N; x, y, z) = self.primal_intermediate (B, N, l, x, y, z)
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
    def primal_base (self, B, N, l, x, y, z):
        delta_x_l = 1

        tmp_mat = np.concatenate((np.concatenate ( (self.H[B, B], self.A[B, :].T), axis = 1), np.concatenate ( (self.A[B, :], -self.M ), axis = 1)), axis = 0 ) #not sure about this whole thing
        tmp_sol = - tmp_mat.I * np.concatenate(self.H[B, l], self.A[:, l]) #wait no sta cosa non mi torna perché sono scalari aaaaaaaaa
        #tutta sta parte del sistema è precaria e sadda provà
        delta_y = tmp_sol[-B.len:] #
        delta_x_B = tmp_sol[:B.len]
    
        delta_z_N = self.H[N, l]*delta_x_l + self.H[B, N].T*delta_x_B - self.A[N].T*delta_y
        delta_z_l = self.H[l, l]*delta_x_l + self.H[B, l].T*delta_x_B - self.A[l].T*delta_y
    
        alpha_opt = math.inf if delta_z_l == 0 else - (z[l] + r[l])/delta_z_l

        # da fare che qui calcoli solo se delta_x è negativo. Però noi abbiamo che non sappiamo delta_x, sooooo
        #nell'intermiedate ho proposto una soluzione ma sono poco sicuro
        alpha_max = np.min((x[B] + self.q[B])/-delta_x_B)
        k = np.argmin((x[B] + self.q[B])/-delta_x_B)

        alpha = min (alpha_opt, alpha_max)

        if (alpha == math.inf):
            return #il problema è impraticabile

        x[l] = x[l] + alpha*delta_x_l
        x[B] = x[B] + alpha*delta_x_b
        y    = y    + alpha*delta_y
        z[l] = z[l] + alpha*delta_z_l
        z[N] = z[N] + alpha*delta_z_N

        if (z[l] + r[l] < 0):
            B[k] = False
            N[k] = True

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
    def primal_intermediate (self, B, N, l, x, y, z):
        delta_z_l = 1

        #robo da risolvere
        #delta_x_l = stuff
        #delta_x_B = stuff
        #delta_y = stuff

        detla_z_N = self.H[N, l] * delta_x_l + self.H.T[B, N] * delta_x_B - self.A.T[N, :] * delta_y
        alpha_opt = -(z[l] + self.r[l])
        min_mask = delta_x_B < 0 # in realtà da rivedere perché questo è di dimensione #B, quindi c'è da aumentare sta maschera o qualcosa del genere
        
        alpha_max = np.min((x[min_mask] + self.q[min_mask])/-delta_x_B[min_mask])
        k = np.argmin(x[min_mask] + self.q[min_mask])/-delt_x_B[min_mask])

        alpha = min(alpha_opt, alpha_max)

        x[l] = x[l] + alpha * delta_l
        x[B] = x[B] + alpha * delta_x_B
        y    = y    + alpha * delta_y
        z[l] = z[l] + alpha * delta_z_l
        z[N] = z[N] + alpha * delta_z_N
        if (z[l] + self.r[l] < 0):
            B[k] = False
            N[k] = True
        return (B, N, x, y, z)
    #kind of a mess but ok
