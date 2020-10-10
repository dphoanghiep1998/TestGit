# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:21:27 2020

Solving Linear algebraic systems

1. Gaussian elimination, LU, PLU, Cholesky decompositions.
2. Iterative methods: Jacobi, Gauss-Seidel
 
@author: phiha
"""

# condtion number of the matrix A
def kA(A):   
    ivA = la.inv(A)  # Find the inverse of a matrix A
    value = la.norm(A,2) * la.norm(ivA,2)    
    return value             

import numpy as np    
import numpy.linalg as la  # import the module linear algebra in numpy
from scipy.linalg import lu, cholesky

#A = np.array([[1, 0],[0, 1]])
#A = np.eye(10)

A = np.random.rand(6,6)
kA(A)


import scipy.linalg as sla
A = sla.hilbert(10)
P,L,U = lu(A)   
print('P is: ',P)
print('L is: ',L)
print('U is: ',U)    

#import time
#print("Pause for some sec")
#time.sleep(5.5)    # pause 5.5 seconds

A = sla.hilbert(10)
print('Phân tích Cholesky trong scipy.linalg cho 1 ma trận tam giác trên U1 thỏa mãn U1.T * U1 = A')
U1 = cholesky(A)
print('U1 is: ',U1)
print('Chuyển vị của U1 là: U1.T = ',U1.T)