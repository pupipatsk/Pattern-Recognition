import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class SimpleBayesClassifier:

    def __init__(self, n_pos, n_neg):
        
        """
        Initializes the SimpleBayesClassifier with prior probabilities.

        Parameters:
        n_pos (int): The number of positive samples.
        n_neg (int): The number of negative samples.
        
        Returns:
        None: This method does not return anything as it is a constructor.
        """

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.prior_pos = n_pos / (n_pos+n_neg)
        self.prior_neg = n_neg / (n_pos+n_neg)

    def fit_params(self, x, y, n_bins = 10):

        """
        Computes histogram-based parameters for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.
        n_bins (int): Number of bins to use for histogram calculation.

        Returns:
        (stay_params, leave_params): A tuple containing two lists of tuples, 
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the bins and edges of the histogram for a feature.
        """

        self.stay_params = [(None, None) for _ in range(x.shape[1])] # class 0
        self.leave_params = [(None, None) for _ in range(x.shape[1])] # class 1
        # INSERT CODE HERE
        for col in range(x.shape[1]):
            hist_stay, bin_edges_stay = np.histogram(x[y == 0, col], bins=n_bins)
            hist_leave, bin_edges_leave = np.histogram(x[y == 1, col], bins=n_bins)

            self.stay_params[col] = (hist_stay, bin_edges_stay)
            self.leave_params[col] = (hist_leave, bin_edges_leave)
        
        return self.stay_params, self.leave_params

    
    def predict(self, x, thresh = 0):
        
        """
        Predicts the class labels for the given samples using the non-parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """
        
        y_pred = []

        return y_pred
    
    def fit_gaussian_params(self, x, y):

        """
        Computes mean and standard deviation for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.

        Returns:
        (gaussian_stay_params, gaussian_leave_params): A tuple containing two lists of tuples,
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the mean and standard deviation for a feature.
        """

        self.gaussian_stay_params = []
        self.gaussian_leave_params = []

        # INSERT CODE HERE
        for col in range(x.shape[1]):
            stay_params = (np.nanmean(x[y == 0, col]), np.nanstd(x[y == 0, col]))
            leave_params = (np.nanmean(x[y == 1, col]), np.nanstd(x[y == 1, col]))
            self.gaussian_stay_params.append(stay_params)
            self.gaussian_leave_params.append(leave_params)
                
        return self.gaussian_stay_params, self.gaussian_leave_params
    
    def gaussian_predict(self, x, thresh = 0):

        """
        Predicts the class labels for the given samples using the parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        for row in range(x.shape[0]):
            lH = np.log(self.prior_pos) - np.log(self.prior_neg) # + sigma(...)
            for col in range(x.shape[1]):
                stay_param = self.gaussian_stay_params[col] # (mu_stay, sd_stay)
                leave_param = self.gaussian_leave_params[col] # (mu_leave, sd_leave)
                
                llh_stay = np.log(stats.norm(stay_param[0],stay_param[1]).pdf(x[row][col]))
                llh_leave = np.log(stats.norm(leave_param[0],leave_param[1]).pdf(x[row][col]))
                
                lH += (llh_leave - llh_stay)
             
            y_pred.append(lH)   
            # # Make a prediction
            # if lH > thresh:
            #     print('lH',lH)
            #     y_pred.append(1)  # Leave
            # else:
            #     print('lH',lH)
            #     y_pred.append(0)  # Stay

        return y_pred