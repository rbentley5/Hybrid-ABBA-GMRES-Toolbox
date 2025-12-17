import numpy as np
from scipy import linalg as la
from trips.utilities.reg_param.gcv import *
from trips.utilities.reg_param.l_curve import l_curve
from pylops import Identity
import time
import GPUtil
from utils import *

def hybrid_BA_GMRES (A, B, b, iter, m, n, num_angles, p = 0, regparam = 'lcurve', stop_rule = 'no', tau = 1.02, x0 = 0, **kwargs):
    
    
    start_time = time.time()
    print("\nHybrid-BA-GMRES is running")

    delta = kwargs['delta'] if ('delta' in kwargs) else None


    if (stop_rule == 'dp') and delta == None:
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

    lambdah_values = []
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
<<<<<<< Updated upstream
                Q_A, R_A, _ = la.svd(H, full_matrices=False)
                R_A = np.diag(R_A)
                R_L = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(Q_A, R_A, R_L, (beta * e), **kwargs)

            elif regparam == 'lcurve':
                lambdah = lcurve(H, (e *beta))
            # elif regparam == 'ncp':
                 
            #lambdah_values[k-1] = lambdah  # Keep track of all computed values
            y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*np.identity(H.shape[1], dtype = 'float32'))),
                                np.vstack(( (beta * e).reshape(-1, 1), np.zeros((H.shape[1], 1), dtype = 'float32') )), rcond=None)[0]
=======
                I = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(U, np.diag(s), I, (beta * e), **kwargs)
            elif regparam == 'lcurve':
                lambdah = (lcurve(s, Vt, c))
            # elif regparam == 'dp':
            #     adj_delta = delta * np.sqrt(k/m)      
            #     lambdah = discrepancy_principle(W[:,:k+1], H, Identity(H.shape[1], H.shape[1]), b.reshape(-1, 1), **kwargs)
            
            lambdah_values.append(lambdah)
            
            #Solve tikhonov regularized problem
            filt = s**2 / (s**2 + lambdah**2)
            y = Vt.T @ ( filt*(c/s))            
>>>>>>> Stashed changes


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
                    if cur_res > res_old or abs(res_old - cur_res)/res_old < 1e-2:  
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
    if ~isinstance(x0, np.ndarray):
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
    lambdah_values = []

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
<<<<<<< Updated upstream
                Q_A, R_A, _ = la.svd(H, full_matrices=False)
                R_A = np.diag(R_A)
                R_L = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(Q_A, R_A, R_L, (beta * e), **kwargs)
            elif regparam == 'lcurve':
                lambdah = lcurve(H, (e *beta))
                 
            #lambdah_values[k-1] = lambdah  # Keep track of all computed values
            y = np.linalg.lstsq(np.vstack((H, np.sqrt(lambdah)*np.identity(H.shape[1], dtype = 'float32'))),
                                np.vstack(( (beta * e).reshape(-1, 1), np.zeros((H.shape[1], 1), dtype = 'float32') )), rcond=None)[0]
=======
                I = Identity(H.shape[1])
                lambdah = generalized_crossvalidation(U, np.diag(s), I, (beta * e), **kwargs)
            elif regparam == 'lcurve':
                lambdah = (lcurve(s, Vt, c))     
            lambdah_values.append(lambdah)  # Keep track of all computed values
            
            filt = s**2 / (s**2 + lambdah**2)
            y = Vt.T @ ( filt*(c/s))             
>>>>>>> Stashed changes

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
                    return X, R, lambdah_values
            
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
                        return X, R, lambdah_values
                    else:
                        Nk_old = Nk
            elif stop_rule == 'RNS':
                cur_res = np.linalg.norm(R[:,k-1])
                if l == 0 and k == 1:
                    res_old = cur_res
                else:
                    if cur_res > res_old or abs(res_old - cur_res)/res_old < 1e-2:  
                        X[:,l*p+1:l*p+k+1] = Xp[:,:k]
                        X = X[:,:l*p + k+1]
                        R = R[:,:l*p + k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Hybrid AB-GMRES execution time: {elapsed_time:.4f} seconds")
                        return X, R, lambdah_values
                    res_old = cur_res

        x0 = Xp[:,k-1]
        r0 = R[:,k-1]
        X[:,l*p+1:l*p+k+1] = Xp
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Hybrid AB-GMRES execution time: {elapsed_time:.4f} seconds")

    return X, R, lambdah_values
def AB_lsqr(A, B, b, iter, m, n, num_angles, stop_rule = 'no', tau = 1.02, **kwargs) :
    start_time = time.time()
    delta = kwargs['delta'] if ('delta' in kwargs) else None

    print("\nAB-lsqr is running")
    
    beta = np.linalg.norm(b)
    u = b / beta
    v = A @ (B @ u)
    alpha = np.linalg.norm(v)
    v = v / alpha
    w = v.copy()
    phi_bar = beta
    rho_bar = alpha

    X = np.zeros((n,iter+1), dtype='float32')
    R = np.zeros((m,iter), dtype='float32')


    for k in range(1, iter+1):
        print("iteration", str(k), "out of",str(iter),end="\r")

        u = A@(B@v) - alpha * u
        beta = np.linalg.norm(u)
        u = u / beta

        v = A@(B@u) - beta * v
        alpha = np.linalg.norm(v)
        v = v / alpha
        rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho 
        theta = s*alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        X[:,k] = X[:,k-1] + (B@((phi / rho)* w)).reshape(-1)
        w = v - (theta/rho)*w
        R[:,k-1] = b - A @ X[:,k-1]

        if stop_rule == 'DP':
                if np.linalg.norm(R[:,k-1]) <= tau*delta*np.sqrt(m):
                    X = X[:,:k+1]
                    R = R[:,:k]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"AB-LSQR execution time: {elapsed_time:.4f} seconds")
                    return X, R
            
        elif stop_rule == 'NCP':
                Nk = NCP(R[:,k-1], m, num_angles)
                if k == 1:
                    Nk_old = Nk
                else:
                    #print('different: ',(Nk_old - Nk))
                    if (Nk_old - Nk) < 0:
                        X = X[:,:k+1]
                        R = R[:,:k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"AB-LSQR execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    else:
                        Nk_old = Nk
        elif stop_rule == 'RNS':
                cur_res = np.linalg.norm(R[:,k-1])
                if k == 1:
                    res_old = cur_res
                else:
                    if cur_res > res_old or abs(res_old - cur_res)/res_old < 1e-4:  
                        X = X[:,:k+1]
                        R = R[:,:k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"AB-LSQR execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    res_old = cur_res
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"AB_lsqr execution time: {elapsed_time:.4f} seconds")

    return X, R
def BA_lsqr(A, B, b, iter, m, n, num_angles, stop_rule = 'no', tau = 1.02, **kwargs) :
    
    delta = kwargs['delta'] if ('delta' in kwargs) else None

    start_time = time.time()
    print("\nBA-lsqr is running")

    
    beta = np.linalg.norm(B@b)
    u = B@b/beta

    v = B@(A@u)
    alpha = np.linalg.norm(v)
    v = v / alpha

    w = v
    phi_bar = beta
    rho_bar = alpha
    
    X = np.zeros((n,iter+1), dtype='float32')
    R = np.zeros((m,iter), dtype='float32')


    for k in range(1, iter+1):
        print("iteration", str(k), "out of",str(iter),end="\r")

        u = B@(A@v) - alpha * u
        beta = np.linalg.norm(u)
        u = u / beta

        v = B@(A@u) - beta * v
        alpha = np.linalg.norm(v)
        v = v / alpha
        rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho 
        theta = s*alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        X[:,k] = X[:,k-1] + ((phi / rho)* w).reshape(-1)
        w = v - (theta/rho)*w
        R[:,k-1] = b - A @ X[:,k-1]
        if stop_rule == 'DP':
                if np.linalg.norm(R[:,k-1]) <= tau*delta*np.sqrt(m):
                    X = X[:,:k+1]
                    R = R[:,:k]
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"BA-LSQR execution time: {elapsed_time:.4f} seconds")

                    return X, R
            
        elif stop_rule == 'NCP':
                Nk = NCP(R[:,k-1], m, num_angles)
                if k == 1:
                    Nk_old = Nk
                else:
                    #print('different: ',(Nk_old - Nk))
                    if (Nk_old - Nk) < 0:
                        X = X[:,:k+1]
                        R = R[:,:k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"BA-LSQR execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    else:
                        Nk_old = Nk
        elif stop_rule == 'RNS':
                cur_res = np.linalg.norm(R[:,k-1])
                if k == 1:
                    res_old = cur_res
                else:
                    if cur_res > res_old or abs(res_old - cur_res)/res_old < 1e-4:  
                        X = X[:,:k+1]
                        R = R[:,:k]
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"BA-LSQR execution time: {elapsed_time:.4f} seconds")
                        return X, R
                    res_old = cur_res

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"BA-LSQR execution time: {elapsed_time:.4f} seconds")

    return X, R