import numpy as np

from util import NMF_model

def nmf_cal(V,Winit,Hinit=None,tol=1e-4,timelimit=100,maxiter=100,W_trainable=False):
    """
    (W,H) = nmf(V,Winit,Hinit,tol,timelimit,maxiter)
    W,H: output solution
    Winit,Hinit: initial solution
    tol: tolerance for a relative stopping condition
    timelimit: ignore
    maxiter: limit of iterations
    """
    if W_trainable==True:
        Model = NMF_model.NMF(n_components=Winit.shape[1],tol=tol, max_iter=maxiter,H_trainable=True)
        W,H = Model.fit_transform(V.astype(np.float))
    else: 
        Model = NMF_model.NMF(n_components=Winit.shape[1],tol=tol, max_iter=maxiter,H_trainable=False)
        X_nmf = V.T
        H_nmf = Winit.T
        W_nmf,H_nmf = Model.fit_transform(X_nmf.astype(np.float),H = H_nmf.astype(np.float))
        H = W_nmf.T
        W = H_nmf.T
        
    return W,H