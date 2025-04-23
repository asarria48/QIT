import numpy as np
from qutip import *

# Define the computational basis of the Hilbert space of two qubits
ket00 = basis(4, 0)
ket01 = basis(4, 1)
ket10 = basis(4, 2)
ket11 = basis(4, 3)
Basis = [ket00, ket01, ket10, ket11]

# Norm of a ket X
def norm(X):
    return np.sqrt(np.sum(np.abs(X)**2))

# Generate a random ket in the computational basis
def randomket():
    coef = np.random.rand(4) + np.random.rand(4) * 1j
    coef = coef / norm(coef)
    ket = Qobj(np.zeros((4, 1)))

    for i in range(4):
        ket += coef[i] * Basis[i]

    return ket

# Generate a random state (Density operator) as a linear combination of N outer products
def randState(N):
    coef = np.abs(np.random.rand(N))
    coef = coef / np.sum(coef)
    p = Qobj(np.zeros((4, 4)))

    for i in range(N):
        psi = randomket()
        p += coef[i] * psi * psi.dag()
    
    return p

# Compute the purity of a random state
def Purity():
    N = np.random.randint(10)
    p = randState(N)
    print(f"The purity of the state is: {(p * p.dag()).tr()}\n")
    return p

Purity()