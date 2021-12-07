import numpy as np

def give_norm(arr,means=None,stds=None):
    if means is None:
        mean = np.mean((np.mean(arr.data[:,:,0,:],axis=0)),axis=0)
        stds = np.std((np.std(arr.data[:,:,0,:],axis=0)),axis=0)
        arr_norm = (arr-mean) / stds
        return arr_norm
    
    elif means is not None:
         arr_norm = (arr-means) / stds
         return arr_norm,means, stds
        
        
def remove_norm(arr,means,stds):
    arr_not_norm = (arr*stds) + means
    return arr_not_norm
