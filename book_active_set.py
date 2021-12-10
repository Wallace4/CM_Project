#!bin/usr/python3
# coding=utf-8

import numpy as np
import math
import sys

class linear_problem:

    def __init__(self, A, b, B, N, x_tmp):
        m, n = A.shape()
        self.e = np.ones(m)

        self.x = x_tmp
        self.y = np.ones(m)
        self.y[B] = -np.sign(A[B] * x_tmp - b[B])
        self.z = A * x - b
        self.z[B] = abs(self.z[B])
        self.z[N] = np.max(-self.z[N], 0, axis=0)

    def solve(self):
        # fai simplesso e scialla

class min_flow_problem:

    @staticmethod
    def __check_shape(x, dim=None), varname=""):
        if isinstance(x, ndarray):
            if dim == None or x.shape == dim:
                return x
            else:
                raise TypeError(f'<{varname}> has shape <{x.shape}> expected <{dim}>')
        else:
            raise TypeError(f'<{varname}> is not {type({np.ndarray})}')

    def __init__ (self, Q, q, E, b, tol=1e-16):

        self.E = self.__checkshape(E, varname="E")
        m, n = self.E.shape

        self.b = self.__check_shape(b, dim=(m,), varname="b")
        self.Q = self.__check_shape(Q, dim=(n,n), varname="Q")
        self.q = self.__check_shape(q, dim=(n,), varname="q")

        self.x = np.zeros(n)

        self.B = np.full(n, True)
        self.N = np.full(n, False)

    def set_initial_solution(self):
        
