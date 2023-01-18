# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) values of parameters of the model      
      b : (scalar)                 value of parameter of the model 
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    cost_w = np.zeros(w.shape)
    cost_b = 0.

    ### START CODE HERE ### 
    for i in range(m):


        z_w = 0        

        for j in range(n):
            z_w += X[i][j]*w[j]
        z_w += b

    
    #    print(sigmoid(z_w))
        f_yi = sigmoid(z_w)-y[i]


        cost_b += f_yi

    #    print(cost_b)

        for j in range(n):
            f_yxi= f_yi*X[i][j]
            cost_w[j] += f_yxi


    dj_db = cost_b /m
    dj_dw = cost_w*(1/m)
    ### END CODE HERE ###

        
    return dj_db, dj_dw
