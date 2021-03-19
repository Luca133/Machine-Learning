from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.01
iterations = 50

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)

print("Theta_0: ", theta_final[0])
print("Theta_1: ", theta_final[1])
print("Theta_2: ", theta_final[2])

#########################################
# Write your code here
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples

sample1 = [1650, 3]
sample2 = [3000, 4]

normalized_sample1 = [1, (sample1[0] - mean_vec[0, 0]) / std_vec[0, 0], ((sample1[1] - mean_vec[0, 1]) / std_vec[0, 1])]
normalized_sample2 = [1, (sample2[0] - mean_vec[0, 0]) / std_vec[0, 0], ((sample2[1] - mean_vec[0, 1]) / std_vec[0, 1])]

print("Normalized sample 1: ", normalized_sample1)
print("Normalized sample 2: ", normalized_sample2)

estimate1 = ((normalized_sample1[0] * theta_final[0]) + (normalized_sample1[1] * theta_final[1]) + (normalized_sample1[2] * theta_final[2]))
print("Estimated value of 1st house is: ", estimate1)

estimate2 = ((normalized_sample1[0] * theta_final[0]) + (normalized_sample2[1] * theta_final[1]) + (normalized_sample2[2] * theta_final[2]))
print("Estimated value of 2nd house is: ", estimate2)

########################################/
