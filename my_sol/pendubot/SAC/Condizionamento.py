import numpy as np

S = np.array([
    [6.841200e+01, 2.177700e+02, 2.043000e+00, 1.167000e+00],
 [2.177700e+02, 8.447149e+03, 4.977500e+01, 3.468900e+01],
 [2.043000e+00, 4.977500e+01, 5.270000e-01, 3.190000e-01],
 [1.167000e+00, 3.468900e+01, 3.190000e-01, 3.130000e-01]])

print(np.linalg.cond(S))

eigvals, eigvecs = np.linalg.eig(S)

print("eigenvalues:", eigvals)

idx = np.argmin(eigvals)
print("worst direction:", eigvecs[:, idx])