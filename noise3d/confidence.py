import numpy as np
import scipy.stats
from .noise import compute_M_corrected

def dof_of_sigmaDs(T, V, H):
    arr = np.array(
        [
            [T, 0, 0, T   , T  , 0  ,  T],
            [0, V, 0, V   , 0  , V  ,  V],
            [0, 0, H, 0   , H  , H  ,  H],
            [T, V, 0, T*V , T  , V  ,  T*V],
            [T, 0, H, T   , T*H, H  ,  T*H],
            [0, V, H, V   , H  , V*H,  V*H],
            [T, V, H, T*V , T*H, V*H,  T*V*H],
    ])
    #arr = np.array(
    #    [
    #        [T-1, 0, 0, T-1   , T-1  , 0  ,  T-1],
    #        [0, V-1, 0, V-1   , 0  , V-1  ,  V-1],
    #        [0, 0, H-1, 0   , H-1  , H-1  ,  H-1],
    #        [T-1, V-1, 0, T*V-1 , T-1  , V-1  ,  T*V-1],
    #        [T-1, 0, H-1, T-1   , T*H-1, H-1  ,  T*H-1],
    #        [0, V-1, H-1, V-1   , H-1  , VH-1,  VH-1],
    #        [T-1, V-1, H-1, T*V-1 , T*H-1, VH-1,  T*V*H-1],
    #])
    return arr


# Cross Correlation matricies
def cc_t(T, V, H, VH):
    arr = np.array(
        [
            [1, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1], 
    ])
    return arr

def cc_v(T, V, H, VH):
    arr = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 1], 
    ])
    return arr

def cc_h(T, V, H, VH):
    arr = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1], 
    ])
    return arr    


def cc_tv(T, V, H, VH):
    arr = np.array(
        [ 
             [1           ,             0, 0,  T/(T*V  - 1),            1,             0, T/(T*V  - 1) ],
             [0           ,             1, 0, V /(T*V  - 1),            0,             1, V /(T*V  - 1)],
             [0           ,             0, 1,             0,            0,             0, 0            ],
             [T/(T*V  - 1), V /(T*V  - 1), 0,             1, T/(T*V  - 1), V /(T*V  - 1), 1            ],
             [1           ,             0, 0,  T/(T*V  - 1),            1,             0,  T/(T*V  - 1)],
             [0           ,             1, 0, V /(T*V  - 1),            0,             1, V /(T*V  - 1)],
             [T/(T*V  - 1), V /(T*V  - 1), 0,             1, T/(T*V  - 1), V /(T*V  - 1), 1            ],
    ])
    return arr    


def cc_th(T, V, H, VH):
    arr = np.array(
        [
            [1           , 0,             0,            1, T/(H *T - 1) ,              0, T/(H *T - 1) ],
            [0           , 1,             0,            0,            0 ,              0, 0            ],
            [0           , 0,             1,            0, H /(H *T - 1),              1, H /(H *T - 1)],
            [1           , 0,             0,            1, T/(H *T - 1) ,              0,  T/(H *T - 1)],
            [T/(H *T - 1), 0, H /(H *T - 1), T/(H *T - 1),             1,  H /(H *T - 1), 1            ],
            [0           , 0,             1,            0, H /(H *T - 1),              1, H /(H *T - 1)],
            [T/(H *T - 1), 0, H /(H *T - 1), T/(H *T - 1),             1,  H /(H *T - 1), 1            ],
    ])
    return arr    

def cc_vh(T, V, H, VH):
    arr = np.array(
        [
            [1,            0,            0,            0,            0,            0, 0           ],
            [0,            1,            0,            1,            0, V /(VH  - 1), V /(VH  - 1)],
            [0,            0,            1,            0,            1, H /(VH  - 1), H /(VH  - 1)],
            [0,            1,            0,            1,            0, V /(VH  - 1), V /(VH  - 1)],
            [0,            0,            1,            0,            1, H /(VH  - 1), H /(VH  - 1)],
            [0, V /(VH  - 1), H /(VH  - 1), V /(VH  - 1), H /(VH  - 1),            1, 1           ],
            [0, V /(VH  - 1), H /(VH  - 1), V /(VH  - 1), H /(VH  - 1),            1, 1           ],
    ])
    return arr    
    
def cc_tvh(T, V, H, VH):
    arr = np.array([
        
         [1           ,           0, 0          ,  T/(T*V-1)     ,  T/(H*T-1)     , 0           ,  T/(H*T*V-1)   ],
         [0           ,           1, 0          ,  V/(T*V-1)     ,  0             , V/(VH-1)    ,  V/(H*T*V-1)   ],
         [0           ,           0, 1          ,  0             ,  H/(H*T-1)     , H/(VH-1)    ,  H/(H*T*V-1)   ],
         [T/(T*V-1)   , V/(T*V-1)  , 0          ,  1             ,  T/(H*T*V-1)   , V/(H*T*V-1) , (T*V)/(H*T*V-1)],  
         [T/(H*T-1)   ,           0, H/(H*T-1)  ,  T/(H*T*V-1)   ,               1, H/(H*T*V-1) , (H*T)/(H*T*V-1)],
         [0           , V/(VH  - 1), H/(VH-1)   ,  V/(H*T*V-1)   ,  H/(H *T*V-1)  , 1           ,  VH/(H*T*V-1)  ],
         [T/(H *T*V-1), V/(H*T*V-1), H/(H*T*V-1), (T*V)/(H*T*V-1), (H*T)/(H*T*V-1), VH/(H*T*V-1),  1             ],  
    ])
    return arr

    



def compute_CI(proba, T, V, H, VH, var_sigmas):
    
    # M tel que sigmas = M sigmaDs
    M = compute_M_corrected(T, V, H, VH)
    MI = np.linalg.inv(M)
    
    # Matrice of degrees of freedom of sigmaDs
    Nv = dof_of_sigmaDs(T, V, H) - 1
    
    # Scale parameters of sigmaDs
    interm = np.multiply(M, np.tile(var_sigmas, (7,1)))
    scale = np.divide(interm, Nv)
    
    # Variances of sigmaDs
    var_sigmaDs = 2 * np.multiply(np.multiply(scale, scale), Nv)
    
    # matrice de crosscorrelation des sigmaDs
    cct = cc_t(T, V, H, VH)
    ccv = cc_v(T, V, H, VH)
    cch = cc_h(T, V, H, VH)
    cctv = cc_tv(T, V, H, VH)
    ccth = cc_th(T, V, H, VH)
    ccvh = cc_vh(T, V, H, VH)
    cctvh = cc_tvh(T, V, H, VH)
    #cc_list = [cct, ccv, cch, cctv, ccth, ccvh, cctvh]
    
    # 7x7x7
    #correlation_matricies_of_sigmaDs = np.stack(cc_list)
    
#    var_pred_matricies = np.stack((np.multiply(cct,var_sigmaDs[:,0]),
#                                  np.multiply(ccv,var_sigmaDs[:,1]),
#                                  np.multiply(cch,var_sigmaDs[:,2]),
#                                  np.multiply(cctv,var_sigmaDs[:,3]),
#                                  np.multiply(ccth,var_sigmaDs[:,4]),
#                                  np.multiply(ccvh,var_sigmaDs[:,5]),
#                                  np.multiply(cctvh,var_sigmaDs[:,6]))
#                                 )
    var_pred_matricies = np.stack((cct*np.sum(var_sigmaDs[:, 0]),
                                  ccv*np.sum(var_sigmaDs[:, 1]),
                                  cch*np.sum(var_sigmaDs[:, 2]),
                                  cctv*np.sum(var_sigmaDs[:, 3]),
                                  ccth*np.sum(var_sigmaDs[:, 4]),
                                  ccvh*np.sum(var_sigmaDs[:, 5]),
                                  cctvh*np.sum(var_sigmaDs[:, 6]),
                                  ))
    
    # matrices de covariance des sigmaDs
    covariance_matricies_of_sigmaDs = np.sum(var_pred_matricies, axis=2)
    
    # matrice de covariance des sigmas
    covariance_matricies_of_sigmas = MI * covariance_matricies_of_sigmaDs * MI.T
    
    # variance des sigmas
    variances_of_sigmas = np.diagonal(covariance_matricies_of_sigmas)
    
    CI = scipy.stats.norm.interval(proba, scale=variances_of_sigmas)
    
    return interm, scale, Nv, var_sigmaDs, var_pred_matricies, covariance_matricies_of_sigmaDs, covariance_matricies_of_sigmas, variances_of_sigmas, CI