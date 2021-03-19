from load_data_ex1 import *
from gradient_descent import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex1()

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((2,1))
# alpha = 0.001 # Start of good range
# alpha = 0.0235 # End of good range
alpha = 0.01 # Best one
# alpha = 0.0000001 # Very low one
# alpha = 1 # Very high one
iterations = 50


# To see it happening slowly
# alpha = 0.00001
# iterations = 50000000000000

# do plotting
do_plot = True

# run gradient descent
t = gradient_descent(X, y, theta, alpha, iterations, do_plot)
