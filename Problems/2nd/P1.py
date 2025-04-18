import numpy as np
from qutip import *

# Declare the vectors 
psi1 = Qobj([[0], [1], [2j]])
psi2 = Qobj([[5], [-2], [1j]])
psi3 = Qobj([[1+1j], [0], [2]])


# a) Normalize the vectors.
psi1_n = psi1.unit()
psi2_n = psi2.unit()
psi3_n = psi3.unit()

print("Normalized vectors:")
print("1.", psi1_n)
print("2.", psi2_n)
print("3.", psi3_n)


# b) Compute the inner products for psi_1, psi_2, and psi_3.
for i, psi_i in enumerate([psi1_n, psi2_n, psi3_n], 1):
    for j, psi_j in enumerate([psi1_n, psi2_n, psi3_n], 1):
        inner_product = psi_i.dag() * psi_j  

        print(f"⟨{i}|{j}⟩ =", inner_product)  


# b) Compute the outer products for psi_1, psi_2, and psi_3.
for i, psi_i in enumerate([psi1_n, psi2_n, psi3_n], 1):
    for j, psi_j in enumerate([psi1_n, psi2_n, psi3_n], 1):
        outer_product = psi_i * psi_j.dag()  

        print(f"|{i}><{j}| =\n", outer_product)   


# c) Compute A for p1 = 1 and p2 = 2. What are the properties of the matrix A?, compute its eigenvectors and eigenvalues.
p1 = 1
p2 = 2
A = p1*(psi1*psi1.dag()) + p2*(psi2*psi2.dag())

# To extract the properties using Qutip I convert A into a Quantum object
A = Qobj(A)
Dim = A.dims
Shape = A.shape
Type = A.type
Hermiticity = A.isherm      # to prove hermiticity (qutip tool)

I = np.eye(A.shape[0])      # identity matrix with the size of A
Unitarity = (A.dag()*A == A*A.dag()) and (A * A.dag() == Qobj(I))  

Symmetricity = (A == A.trans())    # to prove symmetricity 

Idempotency = (A**2 == A)   # to prove idempotency 

print(A)
print("Is 'A' hermitian?", Hermiticity)
print("Is 'A' unitary?", Unitarity)
print("Is 'A' symmetric?", Symmetricity)
print("Is 'A' idempotent?", Idempotency)

# Eigenvalues and eigenvectors

eigenvalues, eigenvectors = A.eigenstates()
print("Eigenvalues:", eigenvalues)

for i, vec in enumerate(eigenvectors):
    print(f"Eigenvector {i+1}:", vec)


# e) Is A an observable? If it is, what are the measurement results one could observe?
print("Since A is hermitian, it is an observable, and the measurment results one could observe are its eigenvalues: \n", eigenvalues) 


# f) Consider a system in the state ρ = (1/N)*|3><3| where N is chosen such that tr(ρ) = 1 holds. What are the probabilities of obtaining the different measurement result if we measure A on ρ.
# First, if N is rightly chosen so the state is correctly normalized, its value must be 6. Now, we could use projectors 
# to compute the probability of measuring the different eigenvalues. 
# I can create a projector using each eigenvector from the matrix A, as its eigenvalues are non degenerate, there will be 3 projectors
# The projectors are defined like the outer product of the eigenvectors

#rho = (1/6)*(psi3*psi3.dag())   # I declare the state
rho = (psi3_n*psi3_n.dag())
probabilities = []  # an empty list in which I will later save the probabilities
for i, eigenvalues in enumerate(eigenvalues):  # I must compute the probabilitie for each eigenvalue to be measured
    P = eigenvectors[i] * eigenvectors[i].dag()  # definition of the projector 
    prob = (P*rho).tr()
    probabilities.append(prob)  # I put the probabilities inside the list

    print(f"Probability for eigenvalue {eigenvalues:.4f}: {prob:.4f}")