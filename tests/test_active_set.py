import pytest

import numpy as np
from active_set import quadratic_problem


@pytest.fixture
def init_matrices():
    np.random.seed(0)
    n, m = 100, 200
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    c = np.random.rand(n)
    H = np.random.rand(n, n)
    M = np.random.rand(m, m)
    q = np.random.rand(n)
    r = np.random.rand(n)
    return A, b, c, H, M, q, r


def test_constructor(init_matrices):
    A, b, c, H, M, q, r = init_matrices
    assert quadratic_problem(A, b, c, H, M)


def test_constructor2(init_matrices):
    A, b, c, H, M, q, r = init_matrices
    b = c  # b size should be wrong
    with pytest.raises(TypeError, match=r"b has shape \(\d+,\) expected \(\d+,\)"):
        assert quadratic_problem(A, b, c, H, M)
