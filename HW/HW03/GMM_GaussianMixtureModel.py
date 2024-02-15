import numpy as np
import matplotlib.pyplot as plt

# Hint: You can use this function to get gaussian distribution.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, mixture_weight, mean_params, cov_params):
        """
        Initialize GMM.
        """
        # Copy construction values.
        self.mixture_weight = mixture_weight
        self.mean_params = mean_params
        self.cov_params = cov_params

        # Initiailize iteration.
        self.n_iter = 0


    def estimation_step(self, data):
        """
        TODO: Perform estimation step. Then, return w_{n,j} in eq. 1)
        """

        # INSERT CODE HERE
        num_samples = data.shape[0]
        num_mixtures = len(self.mixture_weight)
        
        w = np.zeros((num_samples, num_mixtures))
        for n in range(num_samples):
            for j in range(num_mixtures):
                # mixture_weight * prob 
                w[n, j] = self.mixture_weight[j] * multivariate_normal.pdf(data[n], mean=self.mean_params[j], cov=self.cov_params[j])
            w[n] /= np.sum(w[n]) # /Sigma(prob * mixture_weight)

        return w # nparray: n rows, j columns


    def maximization_step(self, data, w):
        """
        TODO: Perform maximization step.
            (Update parameters in this GMM model.)
        """
        # INSERT CODE HERE
        num_samples = data.shape[0]
        num_features = data.shape[1]
        num_mixtures = len(self.mixture_weight)
        
        for j in range(num_mixtures):
            # Update mixture weight
            self.mixture_weight[j] = np.mean(w[:,j])
            
            # Update mean
            self.mean_params[j] = np.sum(w[:,j].reshape(-1,1) * data, axis=0) / np.sum(w[:,j])
            
            # Update covariance
            cov_sum = np.zeros((num_features, num_features))
            for n in range(num_samples):
                diff = data[n] - self.mean_params[j]
                cov_sum += w[n, j] * np.outer(diff, diff)
            self.cov_params[j] = cov_sum / np.sum(w[:, j])
            
            # Convert to diagonal covariance matrix
            self.cov_params[j] = np.diag(np.diag(self.cov_params[j]))  # Set off-diagonal elements to zero


    def get_log_likelihood(self, data):
        """
        TODO: Compute log likelihood.
        """
        # INSERT CODE HERE
        num_samples = data.shape[0]
        num_mixtures = len(self.mixture_weight)
        
        log_likelihood = 0
        for n in range(num_samples):
            
            likelihood = 0
            for j in range(num_mixtures):
                likelihood += self.mixture_weight[j] * multivariate_normal.pdf(data[n], mean=self.mean_params[j], cov=self.cov_params[j])
            
            log_likelihood += np.log(likelihood) if likelihood!=0 else 1e-20
        
        return log_likelihood


    def print_iteration(self):
        print("m :\n", self.mixture_weight)
        print("mu :\n", self.mean_params)
        print("covariance matrix :\n", self.cov_params)
        print("-------------------------------------------------------------")


    def perform_em_iterations(self, data, num_iterations, display=True):
        """
        Perform estimation & maximization steps with num_iterations.
        Then, return list of log_likelihood from those iterations.
        """
        log_prob_list = []

        # Display initialization.
        if display:
            print("Initialization")
            self.print_iteration()

        for n_iter in range(num_iterations):
            
            # TODO: Perform EM step.

            # INSERT CODE HERE

            # E-step
            w = self.estimation_step(data)

            # M-step
            self.maximization_step(data, w)

            # Calculate log prob.
            log_prob = self.get_log_likelihood(data)
            log_prob_list.append(log_prob)

            # Display each iteration.
            if display:
                print(f"Iteration: {n_iter+1}")
                self.print_iteration()

        return log_prob_list