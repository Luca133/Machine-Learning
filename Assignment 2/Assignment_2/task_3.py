import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(0, len(f1)):
    X_full[i, 0] = f1[i]
    X_full[i, 1] = f2[i]
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here
data_npy_file = 'data/PB_data.npy'
# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# Have two counts to "sort" the two phonemes, with phoneme 1 first then phoneme 2
X_phonemes_1_2 = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 2))
j = 0
l = np.sum(phoneme_id == 1)
for i in range(0, len(X_full)):
    if phoneme_id[i] == 1:
        X_phonemes_1_2[j, 0] = X_full[i, 0]
        X_phonemes_1_2[j, 1] = X_full[i, 1]
        j += 1
    elif phoneme_id[i] == 2:
        X_phonemes_1_2[l, 0] = X_full[i, 0]
        X_phonemes_1_2[l, 1] = X_full[i, 1]
        l += 1

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

# Make array of ground truth
ground_truth = np.zeros(len(X_phonemes_1_2))
for i in range(np.sum(phoneme_id == 1)):
    ground_truth[i] = 1

for i in range(np.sum(phoneme_id == 2)):
    ground_truth[i + np.sum(phoneme_id == 1)] = 2


# Load data files generated in task_2.py
phoneme1_gaussian3 = np.load('data/GMM_params_phoneme_01_k_03.npy', allow_pickle=True)
phoneme2_gaussian3 = np.load('data/GMM_params_phoneme_02_k_03.npy', allow_pickle=True)
phoneme1_gaussian6 = np.load('data/GMM_params_phoneme_01_k_06.npy', allow_pickle=True)
phoneme2_gaussian6 = np.load('data/GMM_params_phoneme_02_k_06.npy', allow_pickle=True)

# Convert data files to lists
phoneme1_gaussian3 = np.ndarray.tolist(phoneme1_gaussian3)
phoneme2_gaussian3 = np.ndarray.tolist(phoneme2_gaussian3)
phoneme1_gaussian6 = np.ndarray.tolist(phoneme1_gaussian6)
phoneme2_gaussian6 = np.ndarray.tolist(phoneme2_gaussian6)

# Get predictions
probabilities_phoneme1_gaussian3 = get_predictions(phoneme1_gaussian3['mu'], phoneme1_gaussian3['s'], phoneme1_gaussian3['p'], X_phonemes_1_2)
probabilities_phoneme2_gaussian3 = get_predictions(phoneme2_gaussian3['mu'], phoneme2_gaussian3['s'], phoneme2_gaussian3['p'], X_phonemes_1_2)
probabilities_phoneme1_gaussian6 = get_predictions(phoneme1_gaussian6['mu'], phoneme1_gaussian6['s'], phoneme1_gaussian6['p'], X_phonemes_1_2)
probabilities_phoneme2_gaussian6 = get_predictions(phoneme2_gaussian6['mu'], phoneme2_gaussian6['s'], phoneme2_gaussian6['p'], X_phonemes_1_2)

# Get best prediction using sum value
probabilities_sum_phoneme1_gaussian3 = np.sum(probabilities_phoneme1_gaussian3, axis=1)
probabilities_sum_phoneme2_gaussian3 = np.sum(probabilities_phoneme2_gaussian3, axis=1)
probabilities_sum_phoneme1_gaussian6 = np.sum(probabilities_phoneme1_gaussian6, axis=1)
probabilities_sum_phoneme2_gaussian6 = np.sum(probabilities_phoneme2_gaussian6, axis=1)

# Loop through and use the highest probability to determine whether a point is phoneme 1 or 2
# Get the predicted phoneme of a point for 3 gaussian
predictions_gaussian3 = np.zeros(len(X_phonemes_1_2))
for i in range(0, len(predictions_gaussian3)):
    if probabilities_sum_phoneme1_gaussian3[i] > probabilities_sum_phoneme2_gaussian3[i]:
        predictions_gaussian3[i] = 1
    else:
        predictions_gaussian3[i] = 2

# Get the predicted phoneme of a point for 6 gaussian
predictions_gaussian6 = np.zeros(len(X_phonemes_1_2))
for i in range(0, len(predictions_gaussian6)):
    if probabilities_sum_phoneme1_gaussian6[i] > probabilities_sum_phoneme2_gaussian6[i]:
        predictions_gaussian6[i] = 1
    else:
        predictions_gaussian6[i] = 2


# Check prediction against ground truth
correct_count_3gaussian = 0
for i in range(0, len(predictions_gaussian3)):
    if predictions_gaussian3[i] == ground_truth[i]:
        correct_count_3gaussian += 1

correct_count_6gaussian = 0
for i in range(0, len(predictions_gaussian6)):
    if predictions_gaussian6[i] == ground_truth[i]:
        correct_count_6gaussian += 1

# Calculate accuracy as a percentage
accuracy_3gaussian = (correct_count_3gaussian/len(predictions_gaussian3)) * 100
accuracy_6gaussian = (correct_count_6gaussian/len(predictions_gaussian6)) * 100

########################################/
# Print accuracy for both 3 & 6 gaussian
print('Accuracy using GMMs with {} components: {:.2f}%'.format(3, accuracy_3gaussian))
print('Accuracy using GMMs with {} components: {:.2f}%'.format(6, accuracy_6gaussian))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()