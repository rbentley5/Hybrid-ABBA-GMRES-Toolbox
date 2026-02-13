import numpy as np
from trips.utilities.reg_param.gcv import *
from pylops import Identity
import time
from utils import *

def hybrid_BA_GMRES (A, B, b, iter, m, n, num_angles, p = 0, regparam = 'lcurve', stop_rule = 'no', tau = 1.02, x0 = 0, **kwargs):
    """
    Hybrid BA–GMRES Solver
    =======================

    Solves the linear inverse problem

        A x = b

    using a *hybrid GMRES method* applied to the right-preconditioned system

        B A x = B b

    where B is typically a regularizing or approximate inverse operator.
    The method builds Krylov subspaces of BA and computes a regularized
    solution at each iteration using projected Tikhonov regularization.

    --------------------------------------------------
    PARAMETERS
    --------------------------------------------------
    A : ndarray or LinearOperator (m×n)
        Foward projector.
    B : ndarray or LinearOperator (n×m)
        Back projector.
    b : ndarray length m
        Right-hand side vector.
    iter : int
        Maximum number of iterations.
    m : int
        Number of rows of A (data dimension).
    n : int
        Number of columns of A (unknown dimension).
    num_angles : int
        Number of angles from ct problem.
    p : int, optional
        Restart parameter.
        p = 0 → no restart.
    regparam : str, optional
        Method for choosing regularization parameter:
            "lcurve"  → L-curve criterion (default)
            "gcv"     → generalized cross-validation
            "dp"      → discrepancy principle
    stop_rule : str, optional
        Stopping criterion:
            "dp"   → discrepancy principle
            "ncp"  → normalized cumulative periodogram
            "rns"  → relative norm stagnation
    tau : float, optional
        Safety factor used for discrepancy principle.
    x0 : ndarray or scalar, optional
        Initial guess. If not provided, zero vector is used.
    --------------------------------------------------
    RETURNS
    --------------------------------------------------
    X : ndarray (n × k)
        Matrix whose columns contain all computed iterates:

            X[:,j] = approximate solution at iteration j

        where k ≤ iter depending on stopping rule.

    R : ndarray (m × k-1)
        Residual vectors:

            R[:,j] = b − A X[:,j]
    """
    
    start_time = time.time()
    print("\nHybrid-BA-GMRES is running")

    delta = kwargs['delta'] if ('delta' in kwargs) else None


    if (stop_rule == 'DP') and delta == None:
        raise Exception("A value for the noise level delta was not provided and the discrepancy principle cannot be applied.")
    
    # Check if GMRES should be restarted
    if p == 0:
        p = iter

    # Check if a starting guess was provided
    if not isinstance(x0, np.ndarray):
        x0 = np.zeros((n,)).astype("float32")

    # Make sure p is a divisor of iter else change iter
    L = np.floor(iter/p).astype(int)
    if np.mod(iter,p) != 0:
        iter = L*p

    # Initializations
    b = np.float32(b)
    X = np.zeros((n,iter+1), dtype='float32')
    X[:,0] = x0
    Xp = np.zeros((n,p), dtype='float32')
    R = np.zeros((m,iter), dtype='float32')

    residual = b - A @ x0

    for l in range(0,L):
        r0 = B @ (residual)
        beta = np.linalg.norm(r0)

        W = np.zeros((n,p+1), dtype='float32')
        W[:,0] = r0/beta # Initialization of the first Krylov subspace vector
        
        # Construct the next Krylov subspace vector and solve the least squares problem
        for k in range(1,p+1):
            print("iteration", str(l*p + k), "out of",str(iter),end="\r")

            H = np.zeros((k+1,k), dtype='float32') # Initialize/expand the Hessenberg matrix

            # Insert the previous values of the Hessenberg matrix
            if k > 1:
                H[:k,:k-1] = h_old
            
            q = B @ (A @ W[:,k-1])
            e = np.zeros((k+1,), dtype='float32')
            e[0] = 1

            # Schmidt orthogonalizing the Krylov subspace vector (modified Gram-Schmidt)
            for i in range(1,k+1):
                H[i-1,k-1] = q.reshape(n,1).T @ W[:,i-1].reshape(n,1)
                q = q - H[i-1,k-1]*W[:,i-1] 
            H[k,k-1] = np.linalg.norm(q)
            W[:,k] = q/H[k,k-1] 

            #SVD of the Hessenberg matrix and solution of the regularized problem
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            rhs = beta * e
            c = U.T @ rhs

            #regularization parameter choice
            if k == 1:
                lambdah = 0
            elif regparam == 'gcv':
                I = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(U, np.diag(s), I, (beta * e), **kwargs)
            elif regparam == 'lcurve':
                lambdah = (lcurve(s, Vt, c))
            
            
            #Solve tikhonov regularized problem
            filt = s**2 / (s**2 + lambdah**2)
            y = Vt.T @ ( filt*(c/s))            


            # The solution x_k and its residual
            Xp[:,k-1] = x0 + (W[:,:k] @ np.float32(y)).reshape(-1)
            R[:,k-1] = b - A @ Xp[:,k-1]
            h_old = H
            
            # Stopping rule goes here
            if stop_rule == 'DP': 
                if np.linalg.norm(R[:,k-1]) <= tau*delta*np.sqrt(m):
                    X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                    X = X[:,:l*p + k+1]
                    R = R[:,:l*p + k]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Hybrid BA-GMRES execution time: {elapsed_time:.4f} seconds")
                    return X, R
            
            elif stop_rule == 'NCP':
                Nk = NCP(R[:,k-1], m, num_angles)
                if l == 0 and k == 1:
                    Nk_old = Nk
                else:
                    if (Nk_old - Nk) < 0:
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Hybrid BA-GMRES execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    else:
                        Nk_old = Nk
            elif stop_rule == 'RNS':
                cur_res = np.linalg.norm(R[:,k-1])
                if l == 0 and k == 1:
                    res_old = cur_res
                else:
                    if abs(res_old - cur_res)/res_old < 1e-3:  
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Hybrid BA-GMRES execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    res_old = cur_res
                    
        # Update for next restarted iteration
        x0 = Xp[:,k-1]
        residual = R[:,k-1]
        X[:,l*p+1:l*p+k+1] = Xp
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Hybrid BA-GMRES execution time: {elapsed_time:.4f} seconds")
    return X, R


def hybrid_AB_GMRES (A, B, b, iter, m, n, num_angles, p = 0, regparam = 'lcurve', stop_rule = 'no', tau = 1.02, x0 = 0, **kwargs):

    """
    Hybrid AB–GMRES Solver
    =======================
    Solves the linear inverse problem
        A x = b
    using a *hybrid GMRES method* applied to the right-preconditioned system
        A B y = b
        x = B y
    where B is typically a regularizing or approximate inverse operator.
    The method builds Krylov subspaces of BA and computes a regularized
    solution at each iteration using projected Tikhonov regularization.
    --------------------------------------------------
    PARAMETERS
    --------------------------------------------------
    A : ndarray or LinearOperator (m×n)
        Foward projector.
    B : ndarray or LinearOperator (n×m)
        Back projector.
    b : ndarray length m
        Right-hand side vector.
    iter : int
        Maximum number of iterations.
    m : int
        Number of rows of A (data dimension).
    n : int
        Number of columns of A (unknown dimension).
    num_angles : int
        Number of angles from ct problem.
    p : int, optional
        Restart parameter.
        p = 0 → no restart.
    regparam : str, optional
        Method for choosing regularization parameter:
            "lcurve"  → L-curve criterion (default)
            "gcv"     → generalized cross-validation
            "dp"      → discrepancy principle
    stop_rule : str, optional
        Stopping criterion:
            "dp"   → discrepancy principle
            "ncp"  → normalized cumulative periodogram
            "rns"  → relative norm stagnation
    tau : float, optional
        Safety factor used for discrepancy principle.
    x0 : ndarray or scalar, optional
        Initial guess. If not provided, zero vector is used.
    --------------------------------------------------
    RETURNS
    --------------------------------------------------
    X : ndarray (n × k)
        Matrix whose columns contain all computed iterates:
            X[:,j] = approximate solution at iteration j
        where k ≤ iter depending on stopping rule.
    R : ndarray (m × k-1)
        Residual vectors:
            R[:,j] = b − A X[:,j]
    """

    start_time = time.time()
    delta = kwargs['delta'] if ('delta' in kwargs) else None


    if (regparam == 'dp' or stop_rule == 'dp') and delta == None:
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv or a different stopping criterion.""")

    print("\nHybrid-AB-GMRES is running")

    # Check if GMRES should be restarted
    if p == 0:
        p = iter

    # Check if a starting guess was provided
    if not isinstance(x0, np.ndarray):
        x0 = np.zeros((n,)).astype("float32")
    
    # Make sure p is a divisor of iter else change iter
    L = np.floor(iter/p).astype(int)
    if np.mod(iter,p) != 0:
        iter = L*p

    X = np.zeros((n,iter+1), dtype='float32')
    X[:,0] = x0
    Xp = np.zeros((n,p), dtype='float32')
    R = np.zeros((m,iter), dtype='float32')

    r0 = b - A @ x0

    for l in range(0,L):
        beta   = np.linalg.norm(r0) 

        W = np.zeros((m,p+1), dtype='float32')    
        W[:,0] = r0/beta # Initialization of the first Krylov subspace vector
        
        # Construct the next Krylov subspace vector and solve the least squares problem
        for k in range(1,p+1):
            print("iteration", str(l*p + k), "out of",str(iter),end="\r")
            
            H = np.zeros((k+1,k),dtype='float32') # Initialize/expand the Hessenberg matrix
            
            # Insert the previous values of the Hessenberg matrix
            if k > 1:
                H[:k,:k-1] = h_old

            q = A @ (B @ W[:,k-1])
            e = np.zeros((k+1,), dtype='float32')
            e[0] = 1

            # Schmidt orthogonalizing the Krylov subspace vector (modified Gram-Schmidt)
            for i in range(1,k+1):
                H[i-1,k-1] = q.reshape(m,1).T @ W[:,i-1].reshape(m,1)
                q = q - H[i-1,k-1]*W[:,i-1] 
            H[k,k-1] = np.linalg.norm(q)
            W[:,k] = q/H[k,k-1] 


            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            rhs = beta * e
            c = U.T @ rhs
            
            if k == 1:
                lambdah = 0
            elif regparam == 'gcv':
                I = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(U, np.diag(s), I, (beta * e), **kwargs)
            elif regparam == 'lcurve':
                lambdah = (lcurve(s, Vt, c))     
            
            filt = s**2 / (s**2 + lambdah**2)
            y = Vt.T @ ( filt*(c/s))             

            # The solution x_k and its residual
            Xp[:,k-1] = x0 + (B @ (W[:,:k] @ np.float32(y))).reshape(-1)
            R[:,k-1] = b - A @ Xp[:,k-1]
            h_old = H
        
            # Stopping rule goes here
            if stop_rule == 'DP':
                if np.linalg.norm(R[:,k-1]) <= tau*delta*np.sqrt(m):
                    X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                    X = X[:,:l*p + k+1]
                    R = R[:,:l*p + k]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Hybrid AB-GMRES execution time: {elapsed_time:.4f} seconds")
                    return X, R
            
            elif stop_rule == 'NCP':
                Nk = NCP(R[:,k-1], m, num_angles)
                if l == 0 and k == 1:
                    Nk_old = Nk
                else:
                    if (Nk_old - Nk) < 0:
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Hybrid AB-GMRES execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    else:
                        Nk_old = Nk
            elif stop_rule == 'RNS':
                cur_res = np.linalg.norm(R[:,k-1])
                if l == 0 and k == 1:
                    res_old = cur_res
                else:
                    if cur_res > res_old or abs(res_old - cur_res)/res_old < 1e-3:  
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Hybrid AB-GMRES execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    res_old = cur_res

        x0 = Xp[:,k-1]
        r0 = R[:,k-1]
        X[:,l*p+1:l*p+k+1] = Xp
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Hybrid AB-GMRES execution time: {elapsed_time:.4f} seconds")

    return X, R



