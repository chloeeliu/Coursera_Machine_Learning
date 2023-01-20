# UNQ_C1
# GRADED FUNCTION: estimate_gaussian

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    
    ### START CODE HERE ### 
    mu = np.zeros(n)
    var_sum = np.zeros(n)
    var = np.zeros(n)

    for i in range(n):
        mu[i] = np.average(X[:,i])

        for j in range(m):
            var_sum[i] += (X[j,i] - mu[i])**2
        var[i] = var_sum[i] / m


    ### END CODE HERE ### 
        
    return mu, var
  
  
  
  
  # UNQ_C2
# GRADED FUNCTION: select_threshold

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        ### START CODE HERE ### 
        tp,fp,fn = 0,0,0
        #prec = 0
        #rec = 0
        
        
        for i in range(len(p_val)):
            if p_val[i] < epsilon and y_val[i] == 1 :
                tp += 1
            if p_val[i] < epsilon and y_val[i] == 0 :
                fp += 1
            if p_val[i] >= epsilon and y_val[i] == 1 :
                fn += 1

        if tp != 0:
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            F1 = (2*prec*rec) / (prec+rec)
        
        ### END CODE HERE ### 
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1
  
  
