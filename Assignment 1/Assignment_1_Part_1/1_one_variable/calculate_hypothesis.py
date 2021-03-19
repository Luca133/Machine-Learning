import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    # hypothesis = 0.0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.

    # hypothesis = X[i, 0] * theta[0] + X[i, 1] * theta[1]

    # More general way to calculate hypothesis
    hypothesis = 0.0
    for j in range(0, len(theta)):
        hypothesis += X[i, j] * theta[j]
    
    #########################################
    
    return hypothesis
