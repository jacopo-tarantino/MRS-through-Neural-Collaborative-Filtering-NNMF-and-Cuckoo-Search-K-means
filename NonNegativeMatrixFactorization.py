import numpy as np
from numpy.random import randn, rand
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.sparse as sps
from nnls import nnlsm_blockpivot as nnlstsq

class NonNegativeMatrixFactorization:    
    def censored_lstsq(A, B, M):
        """Solves least squares problem with missing data in B
        Note: uses a broadcasted solve for speed.
        Args
        ----
        A (ndarray) : m x r matrix
        B (ndarray) : m x n matrix
        M (ndarray) : m x n binary matrix (zeros indicate missing values)
        Returns
        -------
        X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
        """
        np.random.seed(55)
        if A.ndim == 1:
            A = A[:,None]

        # else solve via tensor representation
        rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
        T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
        try:
            # transpose to get r x n
            return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T
        except:
            r = T.shape[1]
            T[:,np.arange(r),np.arange(r)] += 1e-6
            return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T

    def censored_nnlstsq(A, B, M):
        """Solves nonnegative least-squares problem with missing data in B
        Args
        ----
        A (ndarray) : m x r matrix
        B (ndarray) : m x n matrix
        M (ndarray) : m x n binary matrix (zeros indicate missing values)
        Returns
        -------
        X (ndarray) : nonnegative r x n matrix that minimizes norm(M*(AX - B))
        """
        np.random.seed(55)
        if A.ndim == 1:
            A = A[:,None]
        rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
        T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
        X = np.empty((B.shape[1], A.shape[1]))
        for n in range(B.shape[1]):
            X[n] =nnlstsq(T[n], rhs[n], is_input_prod=True)[0].T
        return X.T

    def cv_mf(data, rank, M=None, p_holdout=0.3, nonneg=False, max_iterations = 1000,tolerance = 1e-5 ):
        """Fit PCA or NMF while holding out a fraction of the dataset
        Args
        ----
        Data (ndarray) : n x m matrix
        rank : reduced dimension
        M (ndarray) : n x m binary matrix (zeros indicate missing values)
        p_holdout : percentage of missing values in M
        nonneg : True if data is non negative
        max_iteration : possible max_iterations to do
        tolerance : conditions to break the loop
        Returns
        -------
        U (ndarray) : nonnegative n x rank matrix 
        Vt (ndarray) : nonnegative rank x m matrix 
        test_err (int) : squared error on test data
        train_err (int) : squared error on test data
        """
        np.random.seed(55)
        # choose solver for alternating minimization
        if nonneg:
            solver = NonNegativeMatrixFactorization.censored_nnlstsq
        else:
            solver = NonNegativeMatrixFactorization.censored_lstsq

        # create masking matrix
        if M is None:
            M = np.random.rand(*data.shape) > p_holdout

        # initialize U and V randomly
        if nonneg:
                U = np.random.rand(data.shape[0], rank)
                Vt = np.random.rand(rank, data.shape[1])
        else:
                U = np.random.randn(data.shape[0], rank)
                Vt = np.random.randn(rank, data.shape[1])

        # create a copy of U and V
        prev_U = U.copy()
        prev_Vt = Vt.copy()

        # Store differences at each iteration
        diff_U_list = []

        # fit nmf
        for itr in range(max_iterations):
            # update V
            Vt = solver(prev_U, data, M)
            # update U
            U = solver(Vt.T, data.T, M.T).T
        
            # check convergence
            change_U = np.linalg.norm(U - prev_U, 'fro')
            change_Vt = np.linalg.norm(Vt - prev_Vt, 'fro')
            diff_U_list.append(change_U)

            # if both matrices have converged, exit the loop
            if change_U < tolerance and change_Vt < tolerance:
                break

            # update U and V for the next iteration
            prev_U = U
            prev_Vt = Vt
            
            


        # return result and test/train error
        resid = np.dot(U, Vt) - data
        train_err = np.mean(resid[M]**2)
        test_err = np.mean(resid[~M]**2)
        return U, Vt, resid, train_err, test_err, diff_U_list
    
    def plot_pca(data, rank_range=None, test_data_p=0.3, nonneg=False, max_iteration=1000,t=1e-5):
        np.random.seed(55)
        # create storage lists for test and train error
        train_err, test_err = [], []

        # fit models
        for rnk in range(rank_range):
            tr, te = NonNegativeMatrixFactorization.cv_mf(data, rnk,nonneg=nonneg,p_holdout=test_data_p,max_iterations=max_iteration,tolerance=t)[3:5]
            train_err.append((rnk, tr))
            test_err.append((rnk, te))

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
        ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
        ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
        ax.set_ylabel('Mean Squared Error')
        ax.set_xlabel('Number of Latent Features')
        ax.set_title('NNMF')
        ax.axvline(int(np.argmin([tup[1] for tup in test_err])), color='k', dashes=[2, 2])
        ax.spines['top'].set_visible(True)  # Display upper axis
        ax.spines['right'].set_visible(True)  # Display right axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

        # adjust margins
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # tighten the layout
        fig.tight_layout()


        plt.show()

        return int(np.argmin([tup[1] for tup in test_err]))
    
    def get_best_rank(data, rank_range=None, test_data_p=0.3, nonneg=False, max_iteration=1000,t=1e-5):
        np.random.seed(55)
        # create storage lists for test and train error
        train_err, test_err = [], []

        # fit models
        for rnk in range(rank_range):
            tr, te = NonNegativeMatrixFactorization.cv_mf(data, rnk,nonneg=nonneg,p_holdout=test_data_p,max_iterations=max_iteration,tolerance=t)[3:]
            train_err.append((rnk, tr))
            test_err.append((rnk, te))
        
        #select best rank
        return int(np.argmin([tup[1] for tup in test_err]))


        

       

    

            
