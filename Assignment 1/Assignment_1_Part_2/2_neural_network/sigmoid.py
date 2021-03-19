import numpy as np

def sigmoid(z):
    
    #output = 0.0
    #########################################
    # Write your code here
    # modify this to return z passed through the sigmoid function

    # if z is scalar then
    if np.isscalar(z):
        output = 1 / (1 + np.exp(-z))
    else:  # z is an array
        output = np.full_like(z, 1)

        for i in range(0, len(z)):
            output[i] = 1 / (1 + np.exp(-z[i]))
    ########################################/
    
    return output
