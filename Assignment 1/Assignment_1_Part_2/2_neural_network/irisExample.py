from load_data_ex1 import *
from return_test_set import *
from normalize_features import *
import numpy as np
from train_scripts import *
from plot_cost import *
import matplotlib.pyplot as plt
from plot_cost_train_test import *

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

X, y = load_data_ex1()

number_of_trials = 100

minimum_cost = np.zeros(number_of_trials)
average_iterations = np.zeros(number_of_trials)

for i in range(0, number_of_trials):
    print("--------------NEW TRIAL------------------")
    print('Performing trial: #{}'.format(i+1))
    # split the dataset into training and test set
    X_train, y_train, X_test, y_test = return_test_set(X, y)

    # Compute mean and std on train set
    # Normalize both train and test set using these mean and std values
    X_train_normalized, mean_vec, std_vec = normalize_features(X_train)
    X_test_normalized = normalize_features(X_test, mean_vec, std_vec)

    hidden_neurons = 5
    learning_rate = 1.0
    iterations = 100
    is_iris = True

    errors, nn, cost_train, cost_test = train(X_train_normalized, y_train, hidden_neurons, iterations, learning_rate, X_test_normalized, y_test, is_iris)

    # Plot the cost during training for all iterations
    # fig, ax1 = plt.subplots()
    # plot_cost(errors, ax1)
    # plot_filename = os.path.join(os.getcwd(), 'figures', 'Iris_cost.png')
    # plt.savefig(plot_filename)
    min_cost = np.min(errors)
    argmin_cost = np.argmin(errors)
    print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))

    #####################################################
    minimum_cost[i] = min_cost
    average_iterations[i] = argmin_cost+1
    #####################################################
    # Plot the train & test cost for all iterations
    # fig, ax1 = plt.subplots()
    # plot_cost_train_test(cost_train, cost_test, ax1)
    # plot_filename = os.path.join(os.getcwd(), 'figures', 'Iris_train_test_cost.png')
    # plt.savefig(plot_filename)

    # enter non-interactive mode of matplotlib, to keep figures open
    # plt.ioff()
    # plt.show()


average_minimum_cost = np.average(minimum_cost)
average_lowest_iteration = np.average(average_iterations)

print("-------------------AVERAGES-----------------------------")
print('Average minimum cost: {:.5f}'.format(average_minimum_cost))
print('Average lowest iteraion: #{}'.format(average_lowest_iteration))
